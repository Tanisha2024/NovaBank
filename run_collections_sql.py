from pathlib import Path
import re
import sys

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "raw" / "loan.csv"
SQL_FILE = ROOT / "sql" / "collections" / "01_collections_analysis.sql"
REPORT_DIR = ROOT / "reports" / "collections" / "sql"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"LendingClub CSV not found: {DATA_FILE}")
if not SQL_FILE.exists():
    raise FileNotFoundError(f"Collections SQL file not found: {SQL_FILE}")

print(f"Loading LendingClub data: {DATA_FILE}")
data = pd.read_csv(DATA_FILE, low_memory=False)
if "loan_status" not in data.columns:
    raise ValueError("Expected column 'loan_status' was not found in loan.csv")
required = {"recoveries", "collection_recovery_fee", "funded_amnt", "grade", "purpose", "issue_d"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Dataset is missing required collections columns: {sorted(missing)}")

sql_text = SQL_FILE.read_text(encoding="utf-8")
parts = re.split(r"(?m)^-- QUERY:\s*([a-zA-Z0-9_]+)\s*$", sql_text)
if len(parts) < 3:
    raise ValueError("No named queries found. Expected '-- QUERY: query_name' markers.")

queries = {}
for i in range(1, len(parts), 2):
    name = parts[i].strip()
    query = parts[i + 1].strip()
    if query:
        queries[name] = query

REPORT_DIR.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(database=":memory:")
con.register("loan_source", data)

print(f"Loaded {len(data):,} rows; executing {len(queries)} collections SQL queries.")
for name, query in queries.items():
    result = con.execute(query).fetchdf()
    output_file = REPORT_DIR / f"collections_{name}.csv"
    result.to_csv(output_file, index=False)
    print(f"\n{name}: saved {output_file.relative_to(ROOT)}")
    print(result.head(12).to_string(index=False))

con.close()
print(f"\nCollections SQL finished. Reports saved under {REPORT_DIR.relative_to(ROOT)}")
