-- NovaBank: descriptive credit-risk SQL (DuckDB)
-- Run from the NovaBank project root.
-- The cohort contains resolved loans only: bad_outcome=1 means Charged Off; 0 means Fully Paid.
-- These are historical associations, not causal effects.

CREATE OR REPLACE VIEW credit_risk_cohort AS
SELECT *
FROM read_csv_auto(
    'data/processed/credit_risk_cohort.csv',
    header = true,
    nullstr = ['NA', 'N/A', '']
);

-- 1. Baseline outcome metrics.
SELECT
    COUNT(*) AS resolved_loans,
    SUM(CASE WHEN TRY_CAST(bad_outcome AS INTEGER) = 1 THEN 1 ELSE 0 END) AS charged_off_loans,
    SUM(CASE WHEN TRY_CAST(bad_outcome AS INTEGER) = 0 THEN 1 ELSE 0 END) AS fully_paid_loans,
    ROUND(100.0 * AVG(TRY_CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM credit_risk_cohort
WHERE TRY_CAST(bad_outcome AS INTEGER) IN (0, 1);

-- 2. Charge-off rate by grade (A to G).
SELECT
    grade,
    COUNT(*) AS loans,
    SUM(CASE WHEN TRY_CAST(bad_outcome AS INTEGER) = 1 THEN 1 ELSE 0 END) AS charge_offs,
    ROUND(100.0 * AVG(TRY_CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM credit_risk_cohort
WHERE grade IS NOT NULL AND TRY_CAST(bad_outcome AS INTEGER) IN (0, 1)
GROUP BY grade
ORDER BY grade;

-- 3. Charge-off rate by issue vintage/year.
SELECT
    EXTRACT(YEAR FROM TRY_CAST(issue_date AS DATE)) AS issue_year,
    COUNT(*) AS loans,
    SUM(CASE WHEN TRY_CAST(bad_outcome AS INTEGER) = 1 THEN 1 ELSE 0 END) AS charge_offs,
    ROUND(100.0 * AVG(TRY_CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM credit_risk_cohort
WHERE TRY_CAST(issue_date AS DATE) IS NOT NULL
  AND TRY_CAST(bad_outcome AS INTEGER) IN (0, 1)
GROUP BY issue_year
ORDER BY issue_year;

-- 4. Charge-off rate by loan purpose. Read segment sizes with the rate.
SELECT
    purpose,
    COUNT(*) AS loans,
    SUM(CASE WHEN TRY_CAST(bad_outcome AS INTEGER) = 1 THEN 1 ELSE 0 END) AS charge_offs,
    ROUND(100.0 * AVG(TRY_CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM credit_risk_cohort
WHERE purpose IS NOT NULL AND TRY_CAST(bad_outcome AS INTEGER) IN (0, 1)
GROUP BY purpose
HAVING COUNT(*) >= 50
ORDER BY charge_off_rate_pct DESC;

-- 5. Charge-off rate by annual-income quintile (five approximately equal-sized groups).
WITH income_ranked AS (
    SELECT
        TRY_CAST(annual_inc AS DOUBLE) AS annual_inc_num,
        TRY_CAST(bad_outcome AS INTEGER) AS bad_outcome,
        NTILE(5) OVER (ORDER BY TRY_CAST(annual_inc AS DOUBLE)) AS income_quintile
    FROM credit_risk_cohort
    WHERE TRY_CAST(annual_inc AS DOUBLE) > 0
      AND TRY_CAST(bad_outcome AS INTEGER) IN (0, 1)
)
SELECT
    income_quintile,
    MIN(annual_inc_num) AS min_annual_income,
    MAX(annual_inc_num) AS max_annual_income,
    COUNT(*) AS loans,
    SUM(CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END) AS charge_offs,
    ROUND(100.0 * AVG(CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM income_ranked
GROUP BY income_quintile
ORDER BY income_quintile;

-- 6. Charge-off rate by debt-to-income (DTI) band.
WITH dti_banded AS (
    SELECT
        CASE
            WHEN TRY_CAST(dti AS DOUBLE) <= 5 THEN '01: <=5'
            WHEN TRY_CAST(dti AS DOUBLE) <= 10 THEN '02: 5-10'
            WHEN TRY_CAST(dti AS DOUBLE) <= 15 THEN '03: 10-15'
            WHEN TRY_CAST(dti AS DOUBLE) <= 20 THEN '04: 15-20'
            WHEN TRY_CAST(dti AS DOUBLE) <= 25 THEN '05: 20-25'
            WHEN TRY_CAST(dti AS DOUBLE) <= 30 THEN '06: 25-30'
            ELSE '07: >30'
        END AS dti_band,
        TRY_CAST(bad_outcome AS INTEGER) AS bad_outcome
    FROM credit_risk_cohort
    WHERE TRY_CAST(dti AS DOUBLE) IS NOT NULL
      AND TRY_CAST(bad_outcome AS INTEGER) IN (0, 1)
)
SELECT
    dti_band,
    COUNT(*) AS loans,
    SUM(CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END) AS charge_offs,
    ROUND(100.0 * AVG(CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM dti_banded
GROUP BY dti_band
ORDER BY dti_band;

-- 7. Charge-off rate by revolving-credit utilization band.
WITH utilization_banded AS (
    SELECT
        CASE
            WHEN TRY_CAST(revol_util AS DOUBLE) <= 20 THEN '01: <=20%'
            WHEN TRY_CAST(revol_util AS DOUBLE) <= 40 THEN '02: 20-40%'
            WHEN TRY_CAST(revol_util AS DOUBLE) <= 60 THEN '03: 40-60%'
            WHEN TRY_CAST(revol_util AS DOUBLE) <= 80 THEN '04: 60-80%'
            ELSE '05: >80%'
        END AS revol_util_band,
        TRY_CAST(bad_outcome AS INTEGER) AS bad_outcome
    FROM credit_risk_cohort
    WHERE TRY_CAST(revol_util AS DOUBLE) IS NOT NULL
      AND TRY_CAST(bad_outcome AS INTEGER) IN (0, 1)
)
SELECT
    revol_util_band,
    COUNT(*) AS loans,
    SUM(CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END) AS charge_offs,
    ROUND(100.0 * AVG(CAST(bad_outcome AS DOUBLE)), 2) AS charge_off_rate_pct
FROM utilization_banded
GROUP BY revol_util_band
ORDER BY revol_util_band;
