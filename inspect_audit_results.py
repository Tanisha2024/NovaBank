from pathlib import Path
import pandas as pd

report_dir = Path("reports/data_audit")

# 1. Find columns that are entirely missing.
missing = pd.read_csv(
    report_dir / "lendingclub_missing_values.csv"
)

fully_missing = missing[
    missing["missing_count"] == 39717
]

print("\n=== COLUMNS WITH 100% MISSING VALUES ===")
print(fully_missing.to_string(index=False))


# 2. Inspect actual loan outcome categories.
loan_status = pd.read_csv(
    report_dir / "lendingclub_loan_status_distribution.csv"
)

print("\n=== LENDINGCLUB LOAN STATUS ===")
print(loan_status.to_string(index=False))


# 3. Inspect fraud labels after the complete dataset is restored.
fraud_file = (
    report_dir / "paysim_isFraud_distribution.csv"
)

if fraud_file.exists():
    fraud = pd.read_csv(fraud_file)

    print("\n=== PAYSIM FRAUD DISTRIBUTION ===")
    print(fraud.to_string(index=False))