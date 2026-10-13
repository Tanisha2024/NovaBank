from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports/collections")

if not REPORT_DIR.exists():
    raise FileNotFoundError(
        f"Collections report folder not found: {REPORT_DIR.resolve()}"
    )

report_names = [
    "collections_overview.csv",
    "collections_by_grade.csv",
    "collections_by_purpose.csv",
    "collections_by_issue_year.csv",
    "recovery_amount_bands.csv",
]

for name in report_names:
    path = REPORT_DIR / name
    print("\n" + "=" * 88)
    print(f"REPORT: {name}")
    print("=" * 88)

    if not path.exists():
        print(f"Missing report: {path.resolve()}")
        continue

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {name}: {exc}")
        continue

    print(f"Rows: {len(df):,}")
    print("Columns:", ", ".join(map(str, df.columns)))
    if df.empty:
        print("No rows in report.")
    else:
        print(df.head(20).to_string(index=False))
