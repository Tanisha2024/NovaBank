from pathlib import Path

import duckdb
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "data_audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Locate dataset files
# --------------------------------------------------

def find_csv(dataset_name: str) -> Path:
    """Find a dataset by its filename."""

    files = list(RAW_DIR.rglob("*.csv"))

    if dataset_name == "lendingclub":
        matches = [
            p for p in files
            if "loan" in p.name.lower()
            or "lendingclub" in p.name.lower()
        ]

    elif dataset_name == "paysim":
        matches = [
            p for p in files
            if "paysim" in p.name.lower()
            or p.name.lower().startswith("ps_")
        ]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(matches) == 0:
        available = [p.name for p in files]

        raise FileNotFoundError(
            f"Could not find {dataset_name}. "
            f"CSV files found: {available}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple files matched {dataset_name}: {matches}. "
            "Keep the intended source file or specify its path."
        )

    return matches[0]


# --------------------------------------------------
# 3. Safely quote SQL identifiers
# --------------------------------------------------

def quote_identifier(name: str) -> str:
    """Protect column names used in SQL queries."""

    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------
# 4. Audit a dataset
# --------------------------------------------------

def audit_dataset(
    connection,
    dataset_name: str,
    csv_path: Path,
    target_column: str
) -> dict:

    view_name = f"source_{dataset_name}"

    # Escape the file path for the SQL statement.
    sql_path = (
        str(csv_path.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )

    # Read the original CSV through DuckDB.
    # This creates a view, not a modified CSV file.
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT *
        FROM read_csv_auto(
            '{sql_path}',
            header = true,
            sample_size = 10000,
            nullstr=['NA','N/A','']
        )
        """
    )

    # Row count
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {view_name}"
    ).fetchone()[0]

    # Column names and inferred data types
    schema = connection.execute(
        f"DESCRIBE SELECT * FROM {view_name}"
    ).fetchdf()

    schema = schema[["column_name", "column_type"]]

    schema.to_csv(
        OUTPUT_DIR / f"{dataset_name}_schema.csv",
        index=False
    )

    columns = schema["column_name"].tolist()

    # Missing values, including empty text fields
    expressions = []

    for column in columns:
        quoted = quote_identifier(column)

        expressions.append(
            f"""
            SUM(
                CASE
                    WHEN {quoted} IS NULL
                      OR TRIM(CAST({quoted} AS VARCHAR)) = ''
                    THEN 1
                    ELSE 0
                END
            ) AS {quoted}
            """
        )

    missing_wide = connection.execute(
        f"""
        SELECT {", ".join(expressions)}
        FROM {view_name}
        """
    ).fetchdf()

    missing_rows = []

    for column in columns:
        count = int(missing_wide.loc[0, column] or 0)

        missing_rows.append({
            "column": column,
            "missing_count": count,
            "missing_pct": round(
                count / row_count * 100, 3
            ) if row_count else 0
        })

    missing_df = pd.DataFrame(missing_rows)
    missing_df = missing_df.sort_values(
        "missing_pct", ascending=False
    )

    missing_df.to_csv(
        OUTPUT_DIR / f"{dataset_name}_missing_values.csv",
        index=False
    )

    # Target distribution
    actual_columns = {
        column.lower(): column
        for column in columns
    }

    target_file = None
    actual_target = actual_columns.get(target_column.lower())

    if actual_target is not None:
        quoted_target = quote_identifier(actual_target)

        distribution = connection.execute(
            f"""
            SELECT
                CAST({quoted_target} AS VARCHAR) AS category,
                COUNT(*) AS record_count
            FROM {view_name}
            GROUP BY 1
            ORDER BY record_count DESC
            """
        ).fetchdf()

        target_file = f"{dataset_name}_{target_column}_distribution.csv"

        distribution.to_csv(
            OUTPUT_DIR / target_file,
            index=False
        )

    summary = {
        "dataset": dataset_name,
        "source_file": csv_path.name,
        "rows": int(row_count),
        "columns": len(columns),
        "target_column_found": actual_target is not None,
        "missing_cells": int(missing_df["missing_count"].sum()),
        "columns_with_missing_values": int(
            (missing_df["missing_count"] > 0).sum()
        )
    }

    print(f"\n--- {dataset_name.upper()} ---")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nHighest missing-value percentages:")
    print(missing_df.head(10).to_string(index=False))

    if not target_file:
        print(f"\nTarget column '{target_column}' was not found.")

    return summary


# --------------------------------------------------
# 5. Run the audit for both datasets
# --------------------------------------------------

def main():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {RAW_DIR}"
        )

    lendingclub_path = find_csv("lendingclub")
    paysim_path = find_csv("paysim")

    connection = duckdb.connect(database=":memory:")

    try:
        summaries = []

        summaries.append(
            audit_dataset(
                connection,
                "lendingclub",
                lendingclub_path,
                "loan_status"
            )
        )

        summaries.append(
            audit_dataset(
                connection,
                "paysim",
                paysim_path,
                "isFraud"
            )
        )

        summary_df = pd.DataFrame(summaries)

        summary_df.to_csv(
            OUTPUT_DIR / "audit_summary.csv",
            index=False
        )

        print("\nAudit finished successfully.")
        print(f"Reports saved in: {OUTPUT_DIR}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()