from pathlib import Path
import pandas as pd
import numpy as np

# Place this script in the NovaBank project root, beside data/, reports/, and src/.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "fraud_detection" / "eda"
OUTPUT_FILE = OUTPUT_DIR / "fraud_model_data.csv"
SUMMARY_FILE = REPORT_DIR / "model_data_split_summary.csv"
CHUNK_SIZE = 200_000

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]


def main():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"PaySim CSV not found: {SOURCE_FILE}\n"
            "Expected location: NovaBank/data/raw/paysim.csv"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # First pass: find the maximum step without loading the whole CSV into memory.
    max_step = 0
    for frame in pd.read_csv(SOURCE_FILE, usecols=["step"], chunksize=CHUNK_SIZE):
        step_values = pd.to_numeric(frame["step"], errors="coerce")
        if step_values.notna().any():
            max_step = max(max_step, int(step_values.max()))

    if max_step < 3:
        raise ValueError(f"Unexpected maximum step ({max_step}); please verify the dataset.")

    # Chronological split by step: first 70% train, next 15% validation, final 15% test.
    train_end = int(max_step * 0.70)
    validation_end = int(max_step * 0.85)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()

    summary = {
        "train": {"transaction_count": 0, "fraud_count": 0, "min_step": None, "max_step": None},
        "validation": {"transaction_count": 0, "fraud_count": 0, "min_step": None, "max_step": None},
        "test": {"transaction_count": 0, "fraud_count": 0, "min_step": None, "max_step": None},
    }

    selected_rows = 0
    selected_fraud = 0
    first_write = True

    for chunk in pd.read_csv(SOURCE_FILE, usecols=REQUIRED_COLUMNS, chunksize=CHUNK_SIZE):
        # This dataset labels fraud only in TRANSFER/CASH_OUT. Restricting the
        # first model to these types avoids an overly easy model that mainly
        # learns that other transaction types have no fraud labels in PaySim.
        chunk = chunk[chunk["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
        if chunk.empty:
            continue

        for column in ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        chunk = chunk.dropna(subset=["step", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"])
        if chunk.empty:
            continue

        chunk["step"] = chunk["step"].astype("int32")
        chunk["isFraud"] = chunk["isFraud"].astype("int8")

        # Features chosen for a pre-transaction screening scenario.
        # We deliberately exclude isFraud (the target), isFlaggedFraud (an
        # existing rule flag), names/IDs, and newbalanceOrig/newbalanceDest
        # because post-transaction balances may not be available before action.
        chunk["log_amount"] = np.log1p(chunk["amount"].clip(lower=0))
        chunk["amount_to_origin_balance_ratio"] = (
            chunk["amount"] / chunk["oldbalanceOrg"].clip(lower=1)
        )
        chunk["amount_to_destination_balance_ratio"] = (
            chunk["amount"] / chunk["oldbalanceDest"].clip(lower=1)
        )
        chunk["origin_balance_covers_amount"] = (
            chunk["oldbalanceOrg"] >= chunk["amount"]
        ).astype("int8")

        conditions = [
            chunk["step"] <= train_end,
            chunk["step"] <= validation_end,
        ]
        chunk["split"] = np.select(conditions, ["train", "validation"], default="test")

        output_columns = [
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
        ]
        prepared = chunk[output_columns]
        prepared.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        selected_rows += len(prepared)
        selected_fraud += int(prepared["isFraud"].sum())
        for split_name in ["train", "validation", "test"]:
            part = prepared[prepared["split"] == split_name]
            if part.empty:
                continue
            item = summary[split_name]
            item["transaction_count"] += len(part)
            item["fraud_count"] += int(part["isFraud"].sum())
            local_min = int(part["step"].min())
            local_max = int(part["step"].max())
            item["min_step"] = local_min if item["min_step"] is None else min(item["min_step"], local_min)
            item["max_step"] = local_max if item["max_step"] is None else max(item["max_step"], local_max)

    summary_rows = []
    for split_name, item in summary.items():
        count = item["transaction_count"]
        fraud_count = item["fraud_count"]
        summary_rows.append({
            "split": split_name,
            "min_step": item["min_step"],
            "max_step": item["max_step"],
            "transaction_count": count,
            "fraud_count": fraud_count,
            "normal_count": count - fraud_count,
            "fraud_rate_pct": round(100 * fraud_count / count, 6) if count else 0,
        })

    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)

    print("Fraud model dataset preparation finished.")
    print(f"Source: {SOURCE_FILE}")
    print(f"Max PaySim step: {max_step}")
    print(f"Training steps: 1 to {train_end}")
    print(f"Validation steps: {train_end + 1} to {validation_end}")
    print(f"Test steps: {validation_end + 1} to {max_step}")
    print(f"Selected TRANSFER/CASH_OUT rows: {selected_rows:,}")
    print(f"Fraud labels in selected rows: {selected_fraud:,}")
    print(f"Prepared dataset: {OUTPUT_FILE}")
    print(f"Split summary: {SUMMARY_FILE}")
    print("\nSplit summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nNote: this model dataset intentionally excludes post-transaction balances and the existing isFlaggedFraud rule flag.")


if __name__ == "__main__":
    main()
