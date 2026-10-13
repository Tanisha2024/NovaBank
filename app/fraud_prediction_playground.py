"""NovaBank fraud prediction playground (experimental PaySim model).

Place this file beside app.py inside NovaBank/app/ and add the navigation
hook described in the accompanying instructions. The model expects the exact
four-feature order used by compare_fraud_step_ablation.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_xgb_no_step.joblib"
METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "fraud_detection"
    / "modeling"
    / "fraud_step_ablation_comparison.csv"
)

# Keep this order identical to FEATURES_WITHOUT_STEP in
# compare_fraud_step_ablation.py.
FEATURES = [
    "is_transfer",
    "log_amount",
    "log_oldbalanceDest",
    "log_amount_to_destination_balance_ratio",
]


@st.cache_resource(show_spinner="Loading the experimental fraud model...")
def _load_fraud_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def _load_validation_threshold(metrics_path: str):
    """Read the threshold selected on validation, not the test set."""
    metrics_file = Path(metrics_path)
    if not metrics_file.is_file():
        return None

    metrics = pd.read_csv(metrics_file)
    required = {"split", "threshold"}
    if not required.issubset(metrics.columns):
        return None

    rows = metrics[
        metrics["split"].astype(str)
        == "validation_at_validation_best_f1_threshold"
    ]
    if rows.empty:
        return None

    threshold = pd.to_numeric(rows.iloc[0]["threshold"], errors="coerce")
    if pd.isna(threshold) or not 0 <= float(threshold) <= 1:
        return None
    return float(threshold)


def _build_model_row(transaction_type: str, amount: float, oldbalance_dest: float) -> pd.DataFrame:
    """Reproduce the no-step model's feature engineering exactly."""
    amount = max(float(amount), 0.0)
    oldbalance_dest = max(float(oldbalance_dest), 0.0)

    ratio = amount / max(oldbalance_dest, 1.0)
    ratio = float(np.clip(ratio, 0.0, 1_000_000.0))

    row = {
        "is_transfer": 1.0 if transaction_type == "TRANSFER" else 0.0,
        "log_amount": float(np.log1p(amount)),
        "log_oldbalanceDest": float(np.log1p(oldbalance_dest)),
        "log_amount_to_destination_balance_ratio": float(np.log1p(ratio)),
    }
    return pd.DataFrame([row], columns=FEATURES, dtype=np.float32)


def render_fraud_prediction_playground() -> None:
    """Render the add-on page from the main NovaBank Streamlit app."""
    st.markdown(
        '<div class="novabank-eyebrow">03A / Experimental scoring</div>',
        unsafe_allow_html=True,
    )
    st.header("Fraud Prediction Playground")
    st.write(
        "Enter a hypothetical TRANSFER or CASH_OUT transaction to see how the saved "
        "no-step XGBoost model scores it. This is a model demonstration, not a live fraud service."
    )
    st.warning(
        "PaySim is synthetic. The score is not a calibrated probability, and a flag does not "
        "prove fraud. A transaction not flagged by this model is not necessarily safe. "
        "Do not use this prototype for real financial decisions."
    )

    if not MODEL_PATH.is_file():
        st.error(f"Fraud model not found: {MODEL_PATH}")
        st.info("Check that models/fraud_xgb_no_step.joblib exists in the NovaBank project.")
        return

    threshold = _load_validation_threshold(str(METRICS_PATH))
    if threshold is None:
        st.error(f"Could not read the validation-selected threshold from {METRICS_PATH}")
        st.info("Run compare_fraud_step_ablation.py and confirm it created fraud_step_ablation_comparison.csv.")
        return

    st.caption(
        f"Current candidate: fraud_xgb_no_step.joblib · Review threshold: {threshold:.6f} "
        "(selected using the validation split)."
    )

    with st.form("fraud_prediction_playground_form"):
        col1, col2 = st.columns(2)
        with col1:
            transaction_type = st.selectbox(
                "Transaction type",
                options=["TRANSFER", "CASH_OUT"],
                help="The current model was trained only on TRANSFER and CASH_OUT transactions.",
            )
            amount = st.number_input(
                "Transaction amount",
                min_value=0.01,
                value=10000.0,
                step=1000.0,
                help="Enter the amount in the dataset's original currency units.",
            )
        with col2:
            oldbalance_dest = st.number_input(
                "Destination account balance before transaction",
                min_value=0.0,
                value=5000.0,
                step=1000.0,
                help="This is the destination account's recorded balance before the transaction.",
            )
            st.number_input(
                "Model review threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(threshold),
                step=0.001,
                format="%.6f",
                disabled=True,
                help="Read from the threshold selected using validation data; not editable here.",
            )

        submitted = st.form_submit_button("Score transaction", type="primary", use_container_width=True)

    if not submitted:
        st.info("Enter the hypothetical transaction details above, then click **Score transaction**.")
        return

    try:
        model = _load_fraud_model(str(MODEL_PATH))
        model_row = _build_model_row(transaction_type, amount, oldbalance_dest)
        # The model was fit on a NumPy array, so pass a NumPy array in the exact
        # four-feature order to avoid feature-name/order ambiguity.
        feature_array = model_row[FEATURES].to_numpy(dtype=np.float32, copy=True)
        score = float(model.predict_proba(feature_array)[0, 1])
    except Exception as exc:  # show a helpful app error without crashing the entire application
        st.error("The model could not score this input. Check the saved model and installed XGBoost version.")
        st.exception(exc)
        return

    flagged = score >= threshold
    st.divider()
    st.subheader("Prediction result")
    result_col, threshold_col = st.columns(2)
    result_col.metric("Uncalibrated model score", f"{score:.4f}")
    threshold_col.metric("Validation-selected threshold", f"{threshold:.4f}")

    if flagged:
        st.warning("**Flagged for analyst review** by the current experimental threshold.")
    else:
        st.success("**Not flagged at the current threshold.** This does not mean the transaction is safe or confirmed legitimate.")

    st.markdown("**Inputs and engineered features used by the model**")
    details = pd.DataFrame(
        [
            {"Input / model feature": "Transaction type", "Value": transaction_type},
            {"Input / model feature": "Transaction amount", "Value": f"{amount:,.2f}"},
            {"Input / model feature": "Destination balance before transaction", "Value": f"{oldbalance_dest:,.2f}"},
            {"Input / model feature": "is_transfer", "Value": int(transaction_type == "TRANSFER")},
            {"Input / model feature": "log_amount", "Value": f"{model_row.iloc[0]['log_amount']:.6f}"},
            {"Input / model feature": "log_oldbalanceDest", "Value": f"{model_row.iloc[0]['log_oldbalanceDest']:.6f}"},
            {
                "Input / model feature": "log_amount_to_destination_balance_ratio",
                "Value": f"{model_row.iloc[0]['log_amount_to_destination_balance_ratio']:.6f}",
            },
        ]
    )
    st.dataframe(details, hide_index=True, use_container_width=True)
    st.caption(
        "This form scores one hypothetical input only. Its true label is unknown, so this interaction "
        "cannot establish whether the prediction is correct. Use the held-out test reports to evaluate model performance."
    )
