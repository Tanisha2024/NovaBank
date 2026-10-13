from pathlib import Path
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed in this environment. Install it with:\n"
        "  .venv\\Scripts\\python.exe -m pip install xgboost"
    ) from exc


# Place this script in the NovaBank project root, beside data/, models/, and reports/.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "fraud_model_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "fraud_detection" / "modeling"
MODEL_FILE = MODEL_DIR / "fraud_xgb_model.joblib"
METRICS_FILE = REPORT_DIR / "fraud_model_metrics.json"
THRESHOLD_FILE = REPORT_DIR / "fraud_threshold_comparison.csv"
IMPORTANCE_FILE = REPORT_DIR / "fraud_feature_importance.csv"
PREDICTIONS_FILE = REPORT_DIR / "fraud_test_predictions.csv"

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
    "origin_balance_covers_amount",
    "isFraud",
    "split",
}


def safe_metric(value):
    """Convert NumPy values to JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def calculate_metrics(y_true, scores, threshold, split_name):
    predictions = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "split": split_name,
        "threshold": float(threshold),
        "transactions": int(len(y_true)),
        "fraud_cases": int(np.sum(y_true)),
        "fraud_rate_pct": round(float(np.mean(y_true) * 100), 6),
        "roc_auc": safe_metric(roc_auc_score(y_true, scores)),
        "average_precision": safe_metric(average_precision_score(y_true, scores)),
        "precision": safe_metric(precision_score(y_true, predictions, zero_division=0)),
        "recall": safe_metric(recall_score(y_true, predictions, zero_division=0)),
        "f1": safe_metric(f1_score(y_true, predictions, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Prepared fraud data not found: {DATA_FILE}\n"
            "Run prepare_fraud_data.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prepared fraud dataset...")
    dtype_map = {
        "step": "int16",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "oldbalanceDest": "float32",
        "log_amount": "float32",
        "amount_to_origin_balance_ratio": "float32",
        "amount_to_destination_balance_ratio": "float32",
        "origin_balance_covers_amount": "int8",
        "isFraud": "int8",
        "split": "category",
    }
    data = pd.read_csv(DATA_FILE, dtype=dtype_map)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Prepared fraud dataset is missing columns: {sorted(missing)}")

    data = data.dropna(subset=["isFraud", "split", "type"]).copy()
    data["isFraud"] = data["isFraud"].astype("int8")

    unexpected_splits = set(data["split"].astype(str).unique()) - {"train", "validation", "test"}
    if unexpected_splits:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected_splits)}")

    # Input features that are available before the transaction is completed.
    # Do not use isFraud (the answer), isFlaggedFraud (existing rule), IDs,
    # or post-transaction balances as predictors.
    data["is_transfer"] = (data["type"].astype(str) == "TRANSFER").astype("float32")
    data["log_oldbalanceOrg"] = np.log1p(data["oldbalanceOrg"].clip(lower=0)).astype("float32")
    data["log_oldbalanceDest"] = np.log1p(data["oldbalanceDest"].clip(lower=0)).astype("float32")
    data["log_amount_to_origin_balance_ratio"] = np.log1p(
        data["amount_to_origin_balance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1_000_000)
    ).astype("float32")
    data["log_amount_to_destination_balance_ratio"] = np.log1p(
        data["amount_to_destination_balance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1_000_000)
    ).astype("float32")

    feature_columns = [
        "step",
        "is_transfer",
        "log_amount",
        "log_oldbalanceOrg",
        "log_oldbalanceDest",
        "log_amount_to_origin_balance_ratio",
        "log_amount_to_destination_balance_ratio",
        "origin_balance_covers_amount",
    ]

    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float32")
    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_columns + ["isFraud", "split"]).copy()

    train = data[data["split"].astype(str) == "train"]
    validation = data[data["split"].astype(str) == "validation"]
    test = data[data["split"].astype(str) == "test"]

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, or test split is empty.")
    for split_name, subset in [("train", train), ("validation", validation), ("test", test)]:
        if subset["isFraud"].nunique() < 2:
            raise ValueError(f"The {split_name} split must contain both classes.")

    X_train = train[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y_train = train["isFraud"].to_numpy(dtype=np.uint8, copy=True)
    X_validation = validation[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y_validation = validation["isFraud"].to_numpy(dtype=np.uint8, copy=True)
    X_test = test[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y_test = test["isFraud"].to_numpy(dtype=np.uint8, copy=True)

    scale_pos_weight = float((len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1))

    print(f"Prepared rows: {len(data):,}")
    print(f"Training: {len(y_train):,} rows; fraud={int(y_train.sum()):,}; rate={y_train.mean() * 100:.4f}%")
    print(f"Validation: {len(y_validation):,} rows; fraud={int(y_validation.sum()):,}; rate={y_validation.mean() * 100:.4f}%")
    print(f"Test: {len(y_test):,} rows; fraud={int(y_test.sum()):,}; rate={y_test.mean() * 100:.4f}%")
    print(f"Scale positive class weight: {scale_pos_weight:.2f}")
    print("Training XGBoost; this may take several minutes...")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        max_bin=128,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=25,
        )

    # Select a decision threshold on validation only; never tune it on test.
    validation_scores = model.predict_proba(X_validation)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_validation, validation_scores)
    if len(thresholds):
        f1_values = 2 * precisions[:-1] * recalls[:-1] / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
        best_index = int(np.nanargmax(f1_values))
        validation_f1_threshold = float(thresholds[best_index])
    else:
        validation_f1_threshold = 0.5

    test_scores = model.predict_proba(X_test)[:, 1]
    threshold_rows = []
    threshold_rows.append(calculate_metrics(y_validation, validation_scores, 0.5, "validation_at_0.50"))
    threshold_rows.append(calculate_metrics(y_validation, validation_scores, validation_f1_threshold, "validation_at_validation_best_f1_threshold"))
    threshold_rows.append(calculate_metrics(y_test, test_scores, 0.5, "test_at_0.50"))
    threshold_rows.append(calculate_metrics(y_test, test_scores, validation_f1_threshold, "test_at_validation_best_f1_threshold"))

    # Ranking utility at fixed review capacities.
    ranked_indices = np.argsort(-test_scores)
    top_rows = []
    for fraction in [0.01, 0.05, 0.10]:
        k = max(1, int(np.ceil(len(y_test) * fraction)))
        selected_indices = ranked_indices[:k]
        true_positives = int(y_test[selected_indices].sum())
        top_rows.append({
            "review_capacity_pct": fraction * 100,
            "transactions_reviewed": k,
            "fraud_cases_captured": true_positives,
            "capture_rate_pct": round(100 * true_positives / max(int(y_test.sum()), 1), 4),
            "precision_pct": round(100 * true_positives / k, 4),
        })

    # Save model and report artifacts for later Streamlit integration.
    joblib.dump(model, MODEL_FILE)

    feature_importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_.astype(float),
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(IMPORTANCE_FILE, index=False)

    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(THRESHOLD_FILE, index=False)

    prediction_output = test[["step", "type", "amount", "isFraud"]].copy()
    prediction_output["fraud_score"] = test_scores.astype(float)
    prediction_output["predicted_fraud_at_0_50"] = (test_scores >= 0.5).astype("int8")
    prediction_output["predicted_fraud_at_validation_threshold"] = (test_scores >= validation_f1_threshold).astype("int8")
    prediction_output.to_csv(PREDICTIONS_FILE, index=False)

    metrics = {
        "project": "NovaBank Fraud Detection Intelligence",
        "dataset": "PaySim synthetic transaction data",
        "target_column": "isFraud",
        "model": "XGBoostClassifier",
        "feature_columns": feature_columns,
        "excluded_from_features": ["isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest", "nameOrig", "nameDest"],
        "scale_pos_weight": scale_pos_weight,
        "validation_best_f1_threshold": validation_f1_threshold,
        "validation_fraud_rate_pct": round(float(y_validation.mean() * 100), 6),
        "test_fraud_rate_pct": round(float(y_test.mean() * 100), 6),
        "test_no_skill_average_precision_baseline": round(float(y_test.mean()), 8),
        "test_no_skill_roc_auc_baseline": 0.5,
        "threshold_metrics": threshold_rows,
        "top_review_capacity_metrics": top_rows,
        "limitations": [
            "PaySim is synthetic data, not real bank transaction records.",
            "Fraud prevalence changes substantially across chronological splits.",
            "The validation-selected threshold may not remain optimal under distribution shift.",
            "Model scores are not calibrated fraud probabilities.",
            "Feature importance describes model behaviour, not causation.",
        ],
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nFraud model training finished.")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved threshold comparison: {THRESHOLD_FILE}")
    print(f"Saved feature importance: {IMPORTANCE_FILE}")
    print(f"Saved test predictions: {PREDICTIONS_FILE}")
    print("\nThreshold metrics:")
    print(threshold_table[["split", "threshold", "average_precision", "roc_auc", "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"]].to_string(index=False))
    print("\nTop review-capacity results:")
    print(pd.DataFrame(top_rows).to_string(index=False))
    print("\nFeature importance:")
    print(feature_importance.to_string(index=False))
    print("\nReminder: compare the held-out test results carefully because fraud prevalence shifts over time.")


if __name__ == "__main__":
    main()
