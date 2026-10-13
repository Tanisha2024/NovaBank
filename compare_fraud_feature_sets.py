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
RESULTS_FILE = REPORT_DIR / "fraud_feature_ablation_comparison.csv"
IMPORTANCE_FILE = REPORT_DIR / "fraud_feature_ablation_importance.csv"


def json_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def get_validation_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y, scores, threshold):
    pred = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": json_number(roc_auc_score(y, scores)),
        "average_precision": json_number(average_precision_score(y, scores)),
        "precision": json_number(precision_score(y, pred, zero_division=0)),
        "recall": json_number(recall_score(y, pred, zero_division=0)),
        "f1": json_number(f1_score(y, pred, zero_division=0)),
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
        raise FileNotFoundError(f"Prepared fraud data not found: {DATA_FILE}")

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
        "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
        "log_amount", "amount_to_origin_balance_ratio",
        "amount_to_destination_balance_ratio", "origin_balance_covers_amount",
        "isFraud", "split",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")

    data = data.dropna(subset=["isFraud", "split", "type"]).copy()
    data["is_transfer"] = (data["type"].astype(str) == "TRANSFER").astype("float32")
    data["log_oldbalanceOrg"] = np.log1p(data["oldbalanceOrg"].clip(lower=0)).astype("float32")
    data["log_oldbalanceDest"] = np.log1p(data["oldbalanceDest"].clip(lower=0)).astype("float32")
    data["log_amount_to_origin_balance_ratio"] = np.log1p(
        data["amount_to_origin_balance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1_000_000)
    ).astype("float32")
    data["log_amount_to_destination_balance_ratio"] = np.log1p(
        data["amount_to_destination_balance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1_000_000)
    ).astype("float32")

    # Scenario A removes only the binary feature that dominated importance.
    # Scenario B removes the entire origin-balance feature family to test reliance
    # on the simulation-specific origin balance relationship.
    feature_sets = {
        "ablate_binary_cover_only": [
            "step", "is_transfer", "log_amount", "log_oldbalanceOrg",
            "log_oldbalanceDest", "log_amount_to_origin_balance_ratio",
            "log_amount_to_destination_balance_ratio",
        ],
        "ablate_origin_balance_family": [
            "step", "is_transfer", "log_amount", "log_oldbalanceDest",
            "log_amount_to_destination_balance_ratio",
        ],
    }

    y_by_split = {}
    X_by_scenario = {}
    for split_name in ["train", "validation", "test"]:
        part = data[data["split"].astype(str) == split_name].copy()
        if part.empty or part["isFraud"].nunique() < 2:
            raise ValueError(f"Split {split_name} is empty or contains only one class.")
        y_by_split[split_name] = part["isFraud"].to_numpy(dtype=np.uint8, copy=True)
        X_by_scenario[split_name] = {}
        for scenario, features in feature_sets.items():
            X_by_scenario[split_name][scenario] = part[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32, copy=True)

    y_train = y_by_split["train"]
    y_val = y_by_split["validation"]
    y_test = y_by_split["test"]
    weight = float((len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1))
    print(f"Rows: {len(data):,}; positive-class weight: {weight:.2f}")
    print("This diagnostic trains two additional models. It may take several minutes.")

    all_rows = []
    importance_rows = []
    for scenario, features in feature_sets.items():
        print(f"\nTraining scenario: {scenario}")
        model = build_model(weight)
        X_train = X_by_scenario["train"][scenario]
        X_val = X_by_scenario["validation"][scenario]
        X_test = X_by_scenario["test"][scenario]
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        val_scores = model.predict_proba(X_val)[:, 1]
        threshold = get_validation_threshold(y_val, val_scores)
        test_scores = model.predict_proba(X_test)[:, 1]

        for split_name, y, scores, use_threshold in [
            ("validation", y_val, val_scores, threshold),
            ("test", y_test, test_scores, threshold),
            ("test_at_0.50", y_test, test_scores, 0.5),
        ]:
            metrics = evaluate(y, scores, use_threshold)
            all_rows.append({"scenario": scenario, "split": split_name, **metrics})

        model_path = MODEL_DIR / f"fraud_xgb_{scenario}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved diagnostic model: {model_path}")
        print(f"Validation-selected threshold: {threshold:.6f}")
        print("Held-out test metrics:", evaluate(y_test, test_scores, threshold))

        for feature, importance in zip(features, model.feature_importances_.astype(float)):
            importance_rows.append({"scenario": scenario, "feature": feature, "importance": importance})

        # Free scenario arrays/model before the next fit where possible.
        del model

    results = pd.DataFrame(all_rows)
    importance = pd.DataFrame(importance_rows).sort_values(["scenario", "importance"], ascending=[True, False])
    results.to_csv(RESULTS_FILE, index=False)
    importance.to_csv(IMPORTANCE_FILE, index=False)

    print("\nFeature-ablation comparison finished.")
    print(f"Metrics report: {RESULTS_FILE}")
    print(f"Importance report: {IMPORTANCE_FILE}")
    print("\nValidation-selected threshold results:")
    print(results[results["split"].isin(["validation", "test"])][[
        "scenario", "split", "threshold", "average_precision", "roc_auc",
        "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"
    ]].to_string(index=False))
    print("\nTop feature importances by scenario:")
    for scenario, group in importance.groupby("scenario", sort=False):
        print(f"\n{scenario}")
        print(group.head(8).to_string(index=False))
    print("\nImportant: compare these results with the original model metrics. PaySim is synthetic; strong results do not establish real-world bank performance.")


if __name__ == "__main__":
    main()
