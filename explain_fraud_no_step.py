from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise SystemExit(
        "SHAP is not installed. Run: .venv\\Scripts\\python.exe -m pip install shap"
    ) from exc

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "fraud_model_data.csv"
MODEL_FILE = ROOT / "models" / "fraud_xgb_no_step.joblib"
METRICS_FILE = ROOT / "reports" / "fraud_detection" / "modeling" / "fraud_step_ablation_comparison.csv"
REPORT_DIR = ROOT / "reports" / "fraud_detection" / "modeling"

FEATURES = [
    "is_transfer",
    "log_amount",
    "log_oldbalanceDest",
    "log_amount_to_destination_balance_ratio",
]


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Prepared data not found: {DATA_FILE}; run prepare_fraud_data.py first.")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"No-step candidate model not found: {MODEL_FILE}; run compare_fraud_step_ablation.py first.")
    if not METRICS_FILE.is_file():
        raise FileNotFoundError(f"Threshold report not found: {METRICS_FILE}; run compare_fraud_step_ablation.py first.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading no-step fraud model and test records...")
    dtype_map = {
        "step": "int16", "type": "category", "amount": "float32",
        "oldbalanceOrg": "float32", "oldbalanceDest": "float32",
        "log_amount": "float32", "amount_to_origin_balance_ratio": "float32",
        "amount_to_destination_balance_ratio": "float32",
        "origin_balance_covers_amount": "int8", "isFraud": "int8", "split": "category",
    }
    data = pd.read_csv(DATA_FILE, dtype=dtype_map)
    data = data.dropna(subset=["isFraud", "split", "type", "amount", "oldbalanceDest"]).copy()
    data["is_transfer"] = (data["type"].astype(str) == "TRANSFER").astype("float32")
    data["log_oldbalanceDest"] = np.log1p(data["oldbalanceDest"].clip(lower=0)).astype("float32")
    data["log_amount_to_destination_balance_ratio"] = np.log1p(
        data["amount_to_destination_balance_ratio"]
        .replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1_000_000)
    ).astype("float32")

    test = data[data["split"].astype(str) == "test"].copy().reset_index(drop=True)
    if test.empty or test["isFraud"].nunique() < 2:
        raise ValueError("The test split is empty or does not contain both fraud labels.")
    missing = set(FEATURES) - set(test.columns)
    if missing:
        raise ValueError(f"Prepared data is missing features: {sorted(missing)}")

    X_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32").to_numpy(copy=True)
    y_test = test["isFraud"].to_numpy(dtype=np.uint8, copy=True)
    model = joblib.load(MODEL_FILE)
    scores = model.predict_proba(X_test)[:, 1]

    metrics = pd.read_csv(METRICS_FILE)
    val_rows = metrics[metrics["split"].astype(str).str.contains("validation_at_validation_best_f1_threshold")]
    if val_rows.empty:
        raise ValueError("Could not find the validation-selected threshold in fraud_step_ablation_comparison.csv")
    threshold = float(val_rows.iloc[0]["threshold"])
    predicted = (scores >= threshold).astype(np.uint8)

    summary = pd.DataFrame([{
        "candidate_model": MODEL_FILE.name,
        "test_rows": int(len(test)),
        "test_fraud_cases": int(y_test.sum()),
        "test_fraud_rate_pct": round(float(y_test.mean() * 100), 6),
        "validation_selected_threshold": threshold,
        "flagged_transactions": int(predicted.sum()),
        "true_positives": int(((predicted == 1) & (y_test == 1)).sum()),
        "false_positives": int(((predicted == 1) & (y_test == 0)).sum()),
        "false_negatives": int(((predicted == 0) & (y_test == 1)).sum()),
        "note": "Synthetic PaySim retrospective evaluation; not validated for real bank transactions.",
    }])
    summary_path = REPORT_DIR / "fraud_no_step_explanation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Preparing SHAP explainer (may take a little while)...")
    explainer = shap.TreeExplainer(model, feature_names=FEATURES)
    rng = np.random.default_rng(42)
    sample_indices = np.sort(rng.choice(len(X_test), size=min(1500, len(X_test)), replace=False))
    X_sample = X_test[sample_indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_explanation = explainer(X_sample)
    shap_values = np.asarray(sample_explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if shap_values.shape != (len(sample_indices), len(FEATURES)):
        raise ValueError(f"Unexpected SHAP array shape {shap_values.shape}.")

    global_importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_shap": np.mean(np.abs(shap_values), axis=0),
        "mean_shap_direction": np.mean(shap_values, axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)
    global_csv = REPORT_DIR / "fraud_no_step_shap_global_importance.csv"
    global_importance.to_csv(global_csv, index=False)

    global_plot = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, plot_type="bar", max_display=len(FEATURES), show=False)
        plt.tight_layout()
        global_plot = REPORT_DIR / "fraud_no_step_shap_global_importance.png"
        plt.savefig(global_plot, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Warning: global SHAP plot could not be saved: {exc}")

    case_specs = [
        ("true_positive", (y_test == 1) & (predicted == 1)),
        ("false_positive", (y_test == 0) & (predicted == 1)),
        ("false_negative", (y_test == 1) & (predicted == 0)),
    ]
    chosen = []
    for case_type, mask in case_specs:
        indices = np.flatnonzero(mask)
        if len(indices):
            chosen.append((case_type, int(indices[np.argmax(scores[indices])])))

    local_rows = []
    saved_plots = []
    if chosen:
        case_indices = [idx for _, idx in chosen]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            case_explanation = explainer(X_test[case_indices])
        case_values = np.asarray(case_explanation.values)
        if case_values.ndim == 3:
            case_values = case_values[:, :, 1]
        base_values = np.asarray(case_explanation.base_values)
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
        else:
            base_values = base_values.reshape(-1)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for pos, (case_type, row_index) in enumerate(chosen):
            for j, feature in enumerate(FEATURES):
                local_rows.append({
                    "case_type": case_type,
                    "test_row_index": row_index,
                    "true_isFraud": int(y_test[row_index]),
                    "predicted_fraud_at_threshold": int(predicted[row_index]),
                    "fraud_score": float(scores[row_index]),
                    "threshold": threshold,
                    "feature": feature,
                    "observed_feature_value": float(X_test[row_index, j]),
                    "shap_contribution": float(case_values[pos, j]),
                })
            try:
                explanation = shap.Explanation(
                    values=case_values[pos], base_values=float(base_values[pos]),
                    data=X_test[row_index], feature_names=FEATURES,
                )
                shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
                plt.tight_layout()
                plot_path = REPORT_DIR / f"fraud_no_step_shap_{case_type}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                saved_plots.append(plot_path)
            except Exception as exc:
                print(f"Warning: waterfall plot for {case_type} could not be saved: {exc}")

    local_csv = REPORT_DIR / "fraud_no_step_shap_case_examples.csv"
    pd.DataFrame(local_rows).to_csv(local_csv, index=False)
    print("\nNo-step fraud candidate SHAP explanation finished.")
    print(f"Model: {MODEL_FILE}")
    print(f"Test summary: {summary_path}")
    print(f"Global SHAP importance: {global_csv}")
    if global_plot:
        print(f"Global SHAP plot: {global_plot}")
    print(f"Case-level SHAP evidence: {local_csv}")
    for path in saved_plots:
        print(f"Case waterfall plot: {path}")
    print("\nCandidate test summary:")
    print(summary.to_string(index=False))
    print("\nGlobal SHAP importance:")
    print(global_importance.to_string(index=False))
    print("\nReminder: SHAP describes model behaviour; it does not prove causation or real-world banking performance.")


if __name__ == "__main__":
    main()
