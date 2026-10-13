from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "collections"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {"loan_status", "recoveries"}


def read_header(path: Path) -> set[str]:
    try:
        return {str(c).strip().lower() for c in pd.read_csv(path, nrows=0).columns}
    except Exception:
        return set()


def find_source() -> Path:
    # Prefer the original LendingClub source in raw data, then any processed
    # table that still contains the needed outcome/recovery columns.
    candidates = []
    for folder in (RAW_DIR, PROCESSED_DIR):
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.csv")))
    # Prefer conventional LendingClub filenames if several files qualify.
    candidates.sort(key=lambda p: (0 if p.name.lower() in {"loan.csv", "lendingclub.csv", "lendingclub_2007_2011.csv"} else 1, str(p).lower()))
    for path in candidates:
        cols = read_header(path)
        if REQUIRED_COLUMNS.issubset(cols):
            return path
    searched = "\n".join(f" - {p}" for p in candidates) or " - No CSV files found in data/raw or data/processed"
    raise FileNotFoundError(
        "Could not find a CSV containing both 'loan_status' and 'recoveries'.\n"
        "Put the original LendingClub loan.csv in data/raw/ and try again.\n"
        f"CSV files checked:\n{searched}"
    )


def safe_numeric(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def percentage(num, denom):
    return float(num / denom * 100) if denom else 0.0


def main():
    source = find_source()
    print(f"Using LendingClub source: {source}")
    data = pd.read_csv(source, low_memory=False)
    data.columns = [str(c).strip().lower() for c in data.columns]

    if not REQUIRED_COLUMNS.issubset(data.columns):
        raise ValueError(f"Missing required columns: {REQUIRED_COLUMNS - set(data.columns)}")

    data["loan_status"] = data["loan_status"].astype("string").str.strip()
    data["recoveries"] = safe_numeric(data, "recoveries")
    data["collection_recovery_fee"] = safe_numeric(data, "collection_recovery_fee")

    # Recovery analysis is retrospective and restricted to charged-off loans.
    charged = data.loc[data["loan_status"].str.casefold().eq("charged off")].copy()
    if charged.empty:
        statuses = data["loan_status"].value_counts(dropna=False).head(20).to_dict()
        raise ValueError(f"No 'Charged Off' records found. Observed statuses: {statuses}")

    charged["recoveries"] = safe_numeric(charged, "recoveries")
    charged["collection_recovery_fee"] = safe_numeric(charged, "collection_recovery_fee")
    charged["net_recovery_after_fee"] = charged["recoveries"] - charged["collection_recovery_fee"]
    charged["has_recovery"] = charged["recoveries"] > 0

    funded = safe_numeric(charged, "funded_amnt", default=np.nan)
    if funded.isna().all() or (funded.fillna(0) <= 0).all():
        funded = safe_numeric(charged, "loan_amnt", default=np.nan)
    charged["funded_amount_for_ratio"] = funded
    charged["recovery_pct_funded"] = np.where(
        charged["funded_amount_for_ratio"] > 0,
        charged["recoveries"] / charged["funded_amount_for_ratio"] * 100,
        np.nan,
    )

    def summarize(group: pd.DataFrame, key_name: str | None = None) -> dict:
        rec = group["recoveries"]
        rec_positive = rec[rec > 0]
        denom = safe_numeric(group, "funded_amount_for_ratio", default=0).clip(lower=0).sum()
        result = {
            "charged_off_loans": int(len(group)),
            "loans_with_recovery": int(group["has_recovery"].sum()),
            "recovery_incidence_pct": round(percentage(int(group["has_recovery"].sum()), len(group)), 4),
            "total_gross_recoveries": round(float(rec.sum()), 2),
            "mean_recovery_all_charged_off": round(float(rec.mean()), 2) if len(group) else 0.0,
            "median_recovery_all_charged_off": round(float(rec.median()), 2) if len(group) else 0.0,
            "mean_recovery_when_positive": round(float(rec_positive.mean()), 2) if len(rec_positive) else 0.0,
            "median_recovery_when_positive": round(float(rec_positive.median()), 2) if len(rec_positive) else 0.0,
            "total_collection_recovery_fee": round(float(group["collection_recovery_fee"].sum()), 2),
            "net_recovery_after_fee": round(float(group["net_recovery_after_fee"].sum()), 2),
            "gross_recovery_pct_of_funded_amount": round(percentage(float(rec.sum()), float(denom)), 4),
        }
        if key_name:
            result = {key_name: key_value, **result}
        return result

    key_value = None
    overview = summarize(charged)
    overview = {
        "source_file": str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source),
        "all_source_rows": int(len(data)),
        "charged_off_loans": int(len(charged)),
        **{k: v for k, v in overview.items() if k != "charged_off_loans"},
        "scope_note": "Retrospective LendingClub 2007-2011 analysis. Recoveries are analysed only among loans labelled Charged Off; associations are descriptive, not causal.",
    }
    pd.DataFrame([overview]).to_csv(REPORT_DIR / "collections_overview.csv", index=False)

    def category_report(column: str, output_name: str):
        if column not in charged.columns:
            print(f"Skipping {column}: not present in source.")
            return
        temp = charged.copy()
        temp[column] = temp[column].astype("string").fillna("Unknown").str.strip().replace("", "Unknown")
        rows = []
        for value, group in temp.groupby(column, dropna=False):
            rows.append({column: str(value), **summarize(group)})
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("charged_off_loans", ascending=False)
        result.to_csv(REPORT_DIR / output_name, index=False)

    category_report("grade", "collections_by_grade.csv")
    category_report("purpose", "collections_by_purpose.csv")

    # Issue vintage is based on issue_date, or legacy LendingClub issue_d.
    if "issue_date" in charged.columns:
        dates = pd.to_datetime(charged["issue_date"], errors="coerce")
    elif "issue_d" in charged.columns:
        dates = pd.to_datetime(charged["issue_d"].astype("string"), format="%b-%y", errors="coerce")
        if dates.notna().sum() == 0:
            dates = pd.to_datetime(charged["issue_d"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=charged.index)
    charged["issue_year"] = dates.dt.year
    if charged["issue_year"].notna().any():
        rows = []
        for year, group in charged.loc[charged["issue_year"].notna()].groupby("issue_year"):
            rows.append({"issue_year": int(year), **summarize(group)})
        pd.DataFrame(rows).sort_values("issue_year").to_csv(REPORT_DIR / "collections_by_issue_year.csv", index=False)
    else:
        print("Skipping vintage report: neither issue_date nor issue_d could be parsed.")

    # Recovery-size distribution includes no-recovery charged-off loans.
    bins = [-np.inf, 0, 100, 1000, 5000, 10000, 50000, np.inf]
    labels = ["00. No recovery", "01. Greater than 0 to 100", "02. 100 to 1,000", "03. 1,000 to 5,000", "04. 5,000 to 10,000", "05. 10,000 to 50,000", "06. More than 50,000"]
    charged["recovery_amount_band"] = pd.cut(charged["recoveries"], bins=bins, labels=labels, right=True, include_lowest=True)
    rows = []
    for band, group in charged.groupby("recovery_amount_band", observed=False):
        rows.append({"recovery_amount_band": str(band), **summarize(group)})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "recovery_amount_bands.csv", index=False)

    # Limited analytical detail table; do not export addresses, employment titles,
    # postal codes, URLs or other unrelated applicant fields.
    preferred = [
        "id", "record_id", "loan_status", "issue_year", "grade", "purpose",
        "loan_amnt", "funded_amnt", "recoveries", "collection_recovery_fee",
        "net_recovery_after_fee", "recovery_pct_funded", "total_pymnt",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "last_pymnt_d",
        "last_pymnt_amnt", "term", "int_rate", "dti"
    ]
    detail_cols = [c for c in preferred if c in charged.columns]
    charged[detail_cols].to_csv(REPORT_DIR / "charged_off_recovery_detail.csv", index=False)

    manifest = {
        "source_file": str(source),
        "source_rows": int(len(data)),
        "charged_off_rows": int(len(charged)),
        "reports_directory": str(REPORT_DIR),
        "reports_created": sorted(p.name for p in REPORT_DIR.glob("*.csv")),
        "limitations": [
            "Historical LendingClub 2007-2011 data; not representative of current Indian lending portfolios.",
            "Recovery analysis is descriptive and restricted to loans labelled Charged Off.",
            "A recorded recovery is not proof of full repayment or successful collections strategy.",
            "Do not use post-origination recovery fields as predictors in the initial credit-risk model."
        ]
    }
    (REPORT_DIR / "collections_run_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nCollections & Recovery analysis finished.")
    print(f"Source rows: {len(data):,}")
    print(f"Charged-off loans analysed: {len(charged):,}")
    print(f"Loans with positive recovery: {int(charged['has_recovery'].sum()):,} ({percentage(int(charged['has_recovery'].sum()), len(charged)):.2f}%)")
    print(f"Total gross recoveries: ${charged['recoveries'].sum():,.2f}")
    print(f"Total collection recovery fees: ${charged['collection_recovery_fee'].sum():,.2f}")
    print(f"Net recoveries after recorded fees: ${charged['net_recovery_after_fee'].sum():,.2f}")
    print(f"Reports saved under: {REPORT_DIR.relative_to(PROJECT_ROOT)}")
    for file_path in sorted(REPORT_DIR.glob("*.csv")):
        print(f" - {file_path.name}")


if __name__ == "__main__":
    main()
