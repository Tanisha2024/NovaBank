
from pathlib import Path
import pandas as pd

report_dir = Path("reports/fraud_detection/eda")

if not report_dir.exists():
    raise FileNotFoundError(
        f"Reports folder not found: {report_dir.resolve()}"
    )

report_files = sorted(report_dir.glob("*.csv"))

if not report_files:
    raise FileNotFoundError(
        f"No CSV reports found in {report_dir.resolve()}"
    )

print(f"Found {len(report_files)} reports\n")

for file_path in report_files:
    print("\n" + "=" * 70)
    print(f"REPORT: {file_path.name}")
    print("=" * 70)

    report = pd.read_csv(file_path)

    print(f"Rows: {len(report):,}")
    print("Columns:", ", ".join(report.columns))

    if report.empty:
        print("This report has no rows.")
    else:
        print(report.head(15).to_string(index=False))