from pathlib import Path
import os
import re

import duckdb
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SQL_FILE = (
    PROJECT_ROOT
    / "sql"
    / "credit_risk"
    / "01_validate_credit_metrics.sql"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "credit_risk"
    / "sql_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not SQL_FILE.exists():
    raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")

# Ensure relative CSV paths in the SQL file resolve from the project root.
os.chdir(PROJECT_ROOT)

sql_text = SQL_FILE.read_text(encoding="utf-8")

# Split the file using labels such as -- QUERY: baseline
parts = re.split(
    r"(?m)^-- QUERY:\s*([A-Za-z0-9_]+)\s*$",
    sql_text
)
query_names = parts[1::2]
print("Detected SQL queries:", query_names)

if len(parts) < 3:
    raise ValueError(
        "No labelled queries found. Add comments like -- QUERY: baseline."
    )

connection = duckdb.connect(database=":memory:")

results = {}

try:
    for i in range(1, len(parts), 2):
        query_name = parts[i]
        query = parts[i + 1].strip().rstrip(";")

        if not query:
            continue

        result = connection.execute(query).fetchdf()
        results[query_name] = result

        output_file = OUTPUT_DIR / f"{query_name}.csv"
        result.to_csv(output_file, index=False)

        print(f"\n{'=' * 12} {query_name.upper()} {'=' * 12}")
        print(result.to_string(index=False))
        print(f"Saved: {output_file}")

finally:
    connection.close()

print("\nSQL validation queries executed.")