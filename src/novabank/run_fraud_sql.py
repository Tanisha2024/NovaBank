from pathlib import Path
import re
import sys

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "DuckDB is not installed in this Python environment. Run:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install duckdb"
    ) from exc

# Find the actual NovaBank project root
SCRIPT_PATH = Path(__file__).resolve()

ROOT_CANDIDATES = [
    SCRIPT_PATH.parent,
    *SCRIPT_PATH.parents,
]

ROOT = next(
    (
        folder
        for folder in ROOT_CANDIDATES
        if (
            folder / "data" / "raw" / "paysim.csv"
        ).is_file()
        and (
            folder / "sql" / "fraud_detection"
            / "01_fraud_eda.sql"
        ).is_file()
    ),
    None,
)

if ROOT is None:
    raise FileNotFoundError(
        "Cannot find the NovaBank project root. "
        "Check that data/raw/paysim.csv and "
        "sql/fraud_detection/01_fraud_eda.sql "
        "exist under your NovaBank folder."
    )
DATA_FILE = ROOT / "data" / "raw" / "paysim.csv"
SQL_FILE = ROOT / "sql" / "fraud_detection" / "01_fraud_eda.sql"
REPORT_DIR = ROOT / "reports" / "fraud_detection" / "eda"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"PaySim CSV not found: {DATA_FILE}")
if not SQL_FILE.exists():
    raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")

sql_text = SQL_FILE.read_text(encoding="utf-8")
# Execute the view creation separately, using the absolute project CSV path.
view_match = re.search(
    r"CREATE\s+OR\s+REPLACE\s+VIEW\s+paysim\s+AS\s+SELECT\s+\*\s+FROM\s+read_csv_auto\([^;]+\);",
    sql_text,
    flags=re.IGNORECASE | re.DOTALL,
)
if not view_match:
    raise ValueError("Could not find the paysim view definition in the SQL file.")

safe_path = str(DATA_FILE.resolve()).replace("\\", "/").replace("'", "''")
view_sql = (
    "CREATE OR REPLACE VIEW paysim AS "
    f"SELECT * FROM read_csv_auto('{safe_path}', header = true);"
)

# Each labelled query is executed and saved separately for later dashboards.
labelled_queries = re.findall(
    r"--\s*QUERY:\s*([A-Za-z0-9_]+)\s*\n(.*?)(?=--\s*QUERY:|\Z)",
    sql_text,
    flags=re.IGNORECASE | re.DOTALL,
)
if not labelled_queries:
    raise ValueError("No labelled queries found. Expected '-- QUERY: query_name' comments.")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(database=":memory:")
try:
    print("Creating a DuckDB view over the PaySim CSV...")
    con.execute(view_sql)

    for label, query in labelled_queries:
        query = query.strip()
        # Keep the label comments out of the SQL passed to DuckDB.
        query = re.sub(r"^--.*?$", "", query, flags=re.MULTILINE).strip()
        if not query:
            continue
        result = con.execute(query).df()
        output_file = REPORT_DIR / f"{label}.csv"
        result.to_csv(output_file, index=False)
        print(f"\n=== {label} ===")
        print(result.to_string(index=False, max_rows=30))
        print(f"Saved: {output_file.relative_to(ROOT)}")
finally:
    con.close()

print("\nFraud EDA SQL finished. Reports saved under reports/fraud_detection/eda/")
