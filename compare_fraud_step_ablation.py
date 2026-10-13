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
        "xgboost is not installed. Run: .venv\\Scripts\\python.exe -m pip install xgboost"
    ) from exc

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "fraud_model_data.csv"
REPORT_DIR = ROOT / "reports" / "fraud_detection" / "modeling"
MODEL_DIR = ROOT / "models"
BASELINE_FILE = REPORT_DIR / "fraud_feature_ablation_comparison.csv"
METRICS_FILE = REPORT_DIR / "fraud_step_ablation_comparison.csv"
IMPORTANCE_FILE = REPORT_DIR / "fraud_step_ablation_importance.csv"
MODEL_FILE = MODEL_DIR / "fraud_xgb_no_step.joblib"

FEATURES_WITHOUT_STEP = [
    "is_transfer",
    "log_amount",
    "log_oldbalanceDest",
    "log_amount_to_destination_balance_ratio",
]


def safe_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def evaluate(y_true, scores, threshold, split_name):
    predictions = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "scenario": "ablate_step",
        "split": split_name,
        "threshold": float(threshold),
        "transactions": int(len(y_true)),
        "fraud_cases": int(np.sum(y_true)),
        "fraud_rate_pct": round(float(np.mean(y_true) * 100), 6),
        "average_precision": safe_number(average_precision_score(y_true, scores)),
        "roc_auc": safe_number(roc_auc_score(y_true, scores)),
        "precision": safe_number(precision_score(y_true, predictions, zero_division=0)),
        "recall": safe_number(recall_score(y_true, predictions, zero_division=0)),
        "f1": safe_number(f1_score(y_true, predictions, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def build_model(scale_pos_weight):
    return XGBClassifier(
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


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Prepared fraud data not found: {DATA_FILE}; run prepare_fraud_data.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prepared fraud dataset...")
    data = pd.read_csv(
        DATA_FILE,
        dtype={
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
        },
    )

    required = {
        "step", "type", "oldbalanceDest", "log_amount",
        "amount_to_destination_balance_ratio", "isFraud", "split",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")

    data = data.dropna(subset=["isFraud", "split", "type"]).copy()
    data["is_transfer"] = (data["type"].astype(str) == "TRANSFER").astype("float32")
    data["log_oldbalanceDest"] = np.log1p(
        data["oldbalanceDest"].clip(lower=0)
    ).astype("float32")
    data["log_amount_to_destination_balance_ratio"] = np.log1p(
        data["amount_to_destination_balance_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=1_000_000)
    ).astype("float32")

    for column in FEATURES_WITHOUT_STEP:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    data[FEATURES_WITHOUT_STEP] = data[FEATURES_WITHOUT_STEP].replace(
        [np.inf, -np.inf], np.nan
    )
    data = data.dropna(subset=FEATURES_WITHOUT_STEP + ["isFraud", "split"])

    split_data = {}
    for split_name in ["train", "validation", "test"]:
        part = data[data["split"].astype(str) == split_name]
        if part.empty or part["isFraud"].nunique() < 2:
            raise ValueError(f"Split {split_name} is empty or lacks both labels.")
        X = part[FEATURES_WITHOUT_STEP].to_numpy(dtype=np.float32, copy=True)
        y = part["isFraud"].to_numpy(dtype=np.uint8, copy=True)
        split_data[split_name] = (part, X, y)

    train, X_train, y_train = split_data["train"]
    validation, X_val, y_val = split_data["validation"]
    test, X_test, y_test = split_data["test"]

    weight = float((len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1))
    print(f"Prepared rows: {len(data):,}")
    print(f"Train: {len(y_train):,}; fraud={int(y_train.sum()):,}; rate={y_train.mean()*100:.4f}%")
    print(f"Validation: {len(y_val):,}; fraud={int(y_val.sum()):,}; rate={y_val.mean()*100:.4f}%")
    print(f"Test: {len(y_test):,}; fraud={int(y_test.sum()):,}; rate={y_test.mean()*100:.4f}%")
    print(f"Positive-class weight: {weight:.2f}")
    print("Training the same XGBoost configuration without the step feature...")

    model = build_model(weight)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]

    rows = [
        evaluate(y_val, val_scores, threshold, "validation_at_validation_best_f1_threshold"),
        evaluate(y_test, test_scores, threshold, "test_at_validation_best_f1_threshold"),
        evaluate(y_test, test_scores, 0.5, "test_at_0.50"),
    ]
    results = pd.DataFrame(rows)
    results.to_csv(METRICS_FILE, index=False)
    joblib.dump(model, MODEL_FILE)

    importance = pd.DataFrame({
        "feature": FEATURES_WITHOUT_STEP,
        "importance": model.feature_importances_.astype(float),
    }).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_FILE, index=False)

    review_rows = []
    order = np.argsort(-test_scores)
    for fraction in [0.01, 0.05, 0.10]:
        k = max(1, int(np.ceil(len(y_test) * fraction)))
        chosen = order[:k]
        found = int(y_test[chosen].sum())
        review_rows.append({
            "review_capacity_pct": fraction * 100,
            "transactions_reviewed": k,
            "fraud_cases_captured": found,
            "capture_rate_pct": round(100 * found / max(int(y_test.sum()), 1), 4),
            "precision_pct": round(100 * found / k, 4),
        })
    review_path = REPORT_DIR / "fraud_step_ablation_review_capacity.csv"
    pd.DataFrame(review_rows).to_csv(review_path, index=False)

    print("\nStep-ablation training finished.")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Metrics saved: {METRICS_FILE}")
    print(f"Feature importance saved: {IMPORTANCE_FILE}")
    print(f"Review-capacity report saved: {review_path}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print("\nMetrics:")
    print(results[["split", "threshold", "average_precision", "roc_auc", "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"]].to_string(index=False))
    print("\nFeature importance without step:")
    print(importance.to_string(index=False))
    print("\nReview capacity without step:")
    print(pd.DataFrame(review_rows).to_string(index=False))

    if BASELINE_FILE.is_file():
        baseline = pd.read_csv(BASELINE_FILE)
        match = baseline[(baseline["scenario"] == "ablate_origin_balance_family") & (baseline["split"] == "test")]
        if not match.empty:
            print("\nPreviously measured candidate WITH step (for reference):")
            print(match[["threshold", "average_precision", "roc_auc", "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"]].to_string(index=False))

    print("\nInterpretation note: this is a controlled feature-ablation experiment on synthetic PaySim data; it does not establish real-world banking performance.")


if __name__ == "__main__":
    main()
