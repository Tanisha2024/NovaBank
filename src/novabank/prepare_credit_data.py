from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "loan.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_audit"

OUTPUT_FILE = OUTPUT_DIR / "credit_risk_cohort.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Define business rules
# --------------------------------------------------

GOOD_STATUS = "Fully Paid"
BAD_STATUS = "Charged Off"

# Initial application-profile predictors.
# Only columns that exist and contain some data will be kept.
CANDIDATE_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "pub_rec_bankruptcies",
    "tax_liens",
    "acc_now_delinq",
    "collections_12_mths_ex_med",
]


NUMERIC_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "pub_rec_bankruptcies",
    "tax_liens",
    "acc_now_delinq",
    "collections_12_mths_ex_med",
]


# --------------------------------------------------
# 3. Load original dataset
# --------------------------------------------------

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the raw dataset: {RAW_FILE}"
    )

loan = pd.read_csv(RAW_FILE, low_memory=False)

required_columns = {"loan_status", "issue_d"}

missing_required = required_columns - set(loan.columns)

if missing_required:
    raise ValueError(
        f"Required columns are missing: {missing_required}"
    )

print("Original dataset shape:", loan.shape)


# --------------------------------------------------
# 4. Validate the observed loan statuses
# --------------------------------------------------

print("\nOriginal loan-status distribution:")
print(loan["loan_status"].value_counts(dropna=False))

observed_statuses = set(
    loan["loan_status"].dropna().unique()
)

allowed_statuses = {GOOD_STATUS, BAD_STATUS, "Current"}

unexpected_statuses = observed_statuses - allowed_statuses

if unexpected_statuses:
    print(
        "\nAdditional statuses require review:",
        unexpected_statuses
    )


# --------------------------------------------------
# 5. Keep loans with observed final outcomes
# --------------------------------------------------

resolved = loan[
    loan["loan_status"].isin([GOOD_STATUS, BAD_STATUS])
].copy()

excluded_count = len(loan) - len(resolved)

if resolved.empty:
    raise ValueError(
        "No fully paid or charged-off loans were found."
    )

# The target is 1 for charged off and 0 for fully paid.
resolved["bad_outcome"] = (
    resolved["loan_status"] == BAD_STATUS
).astype("int8")

print("\nResolved loans:", len(resolved))
print("Excluded unresolved/other records:", excluded_count)


# --------------------------------------------------
# 6. Parse the dates needed for temporal validation
# --------------------------------------------------

resolved["issue_date"] = pd.to_datetime(
    resolved["issue_d"].astype("string").str.strip(),
    format="%b-%y",
    errors="coerce"
)

resolved["issue_year"] = resolved["issue_date"].dt.year

invalid_issue_dates = int(resolved["issue_date"].isna().sum())

print("Unparsed issue dates:", invalid_issue_dates)


# --------------------------------------------------
# 7. Build the credit-history-age feature
# --------------------------------------------------

# Both dates refer to information available at loan issuance.
# This avoids using later repayment information.

if "earliest_cr_line" in resolved.columns:

    earliest_date = pd.to_datetime(
        resolved["earliest_cr_line"]
        .astype("string")
        .str.strip(),
        format="%b-%y",
        errors="coerce"
    )

    history_days = (
        resolved["issue_date"] - earliest_date
    ).dt.days

    resolved["credit_history_months"] = (
        history_days / 30.4375
    )

    # Impossible date sequences are marked missing for review.
    invalid_history = (
        resolved["credit_history_months"] < 0
    )

    resolved.loc[
        invalid_history, "credit_history_months"
    ] = float("nan")


# --------------------------------------------------
# 8. Select available, non-empty candidate features
# --------------------------------------------------

available_features = [
    column
    for column in CANDIDATE_FEATURES
    if (
        column in resolved.columns
        and resolved[column].notna().any()
    )
]

unavailable_features = [
    column
    for column in CANDIDATE_FEATURES
    if column not in available_features
]

print("\nSelected candidate features:")
print(available_features)

print("\nFeatures absent or entirely missing:")
print(unavailable_features)


# --------------------------------------------------
# 9. Convert numeric fields safely
# --------------------------------------------------

for column in available_features:

    if column not in NUMERIC_FEATURES:
        continue

    values = resolved[column].astype("string").str.strip()

    # revol_util may contain percent symbols.
    if column in {'revol_util','int_rate'}:
        values = values.str.replace("%", "", regex=False)

    resolved[column] = pd.to_numeric(
        values,
        errors="coerce"
    )


# --------------------------------------------------
# 10. Preserve categorical fields without encoding yet
# --------------------------------------------------

categorical_features = [
    column
    for column in available_features
    if column not in NUMERIC_FEATURES
]

for column in categorical_features:
    resolved[column] = (
        resolved[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


# --------------------------------------------------
# 11. Construct a modelling and validation dataset
# --------------------------------------------------

# IDs are retained for traceability, never as model predictors.
output_columns = []

if "id" in resolved.columns:
    resolved["record_id"] = resolved["id"].astype("string")
else:
    resolved["record_id"] = resolved.index.astype("string")

output_columns.append("record_id")

# Dates/year are metadata for time-based validation.
# They will not automatically be passed into the model.
output_columns.extend(["issue_date", "issue_year"])

# This feature is derived from dates available at loan issuance.
if "credit_history_months" in resolved.columns:
    output_columns.append("credit_history_months")

output_columns.extend(available_features)
output_columns.append("bad_outcome")

credit_data = resolved[output_columns].copy()

# Defensive checks: target must be binary and contain both classes.
assert set(credit_data["bad_outcome"].unique()) == {0, 1}
assert credit_data["record_id"].notna().all()


# --------------------------------------------------
# 12. Save data and supporting reports
# --------------------------------------------------

credit_data.to_csv(OUTPUT_FILE, index=False)

target_counts = (
    credit_data["bad_outcome"]
    .value_counts()
    .rename_axis("bad_outcome")
    .reset_index(name="loan_count")
)

target_counts["share_pct"] = (
    target_counts["loan_count"]
    / len(credit_data)
    * 100
).round(2)

target_counts.to_csv(
    REPORT_DIR / "credit_target_distribution.csv",
    index=False
)

feature_missingness = pd.DataFrame({
    "feature": available_features,
    "missing_count": [
        int(credit_data[column].isna().sum())
        for column in available_features
    ],
})

feature_missingness["missing_pct"] = (
    feature_missingness["missing_count"]
    / len(credit_data)
    * 100
).round(2)

feature_missingness.sort_values(
    "missing_pct",
    ascending=False
).to_csv(
    REPORT_DIR / "credit_feature_missingness.csv",
    index=False
)

summary = pd.DataFrame([{
    "original_rows": len(loan),
    "resolved_rows": len(credit_data),
    "excluded_rows": excluded_count,
    "charged_off_count": int(
        (credit_data["bad_outcome"] == 1).sum()
    ),
    "fully_paid_count": int(
        (credit_data["bad_outcome"] == 0).sum()
    ),
    "charged_off_rate_pct": round(
        credit_data["bad_outcome"].mean() * 100,
        2
    ),
    "features_selected": len(available_features),
    "unparsed_issue_dates": invalid_issue_dates
}])

summary.to_csv(
    REPORT_DIR / "credit_preparation_summary.csv",
    index=False
)

print("\nCredit-risk dataset prepared successfully.")
print("Output:", OUTPUT_FILE)
print("\nPreparation summary:")
print(summary.to_string(index=False))