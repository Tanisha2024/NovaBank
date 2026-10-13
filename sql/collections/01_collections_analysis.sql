-- NovaBank Collections & Recovery Intelligence
-- Analytical scope: LendingClub loans labelled Charged Off.
-- All results are retrospective descriptive statistics, not causal estimates.
-- The Python runner registers the raw CSV as `loan_source` before executing these queries.

-- QUERY: overview
WITH source AS (
    SELECT
        *,
        COALESCE(TRY_CAST(recoveries AS DOUBLE), 0.0) AS recovery_amount,
        COALESCE(TRY_CAST(collection_recovery_fee AS DOUBLE), 0.0) AS recovery_fee,
        COALESCE(TRY_CAST(funded_amnt AS DOUBLE), 0.0) AS funded_amount
    FROM loan_source
),
charged_off AS (
    SELECT * FROM source WHERE loan_status = 'Charged Off'
)
SELECT
    'data/raw/loan.csv' AS source_file,
    (SELECT COUNT(*) FROM source) AS all_source_rows,
    COUNT(*) AS charged_off_loans,
    COUNT(*) FILTER (WHERE recovery_amount > 0) AS loans_with_recovery,
    ROUND(100.0 * COUNT(*) FILTER (WHERE recovery_amount > 0) / NULLIF(COUNT(*), 0), 2) AS recovery_incidence_pct,
    ROUND(SUM(recovery_amount), 2) AS total_gross_recoveries,
    ROUND(AVG(recovery_amount), 2) AS mean_recovery_all_charged_off,
    ROUND(MEDIAN(recovery_amount), 2) AS median_recovery_all_charged_off,
    ROUND(AVG(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS mean_recovery_when_positive,
    ROUND(MEDIAN(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS median_recovery_when_positive,
    ROUND(SUM(recovery_fee), 2) AS total_collection_recovery_fee,
    ROUND(SUM(recovery_amount - recovery_fee), 2) AS net_recovery_after_fee,
    ROUND(100.0 * SUM(recovery_amount) / NULLIF(SUM(funded_amount), 0), 4) AS gross_recovery_pct_of_funded_amount,
    'Retrospective LendingClub 2007-2011 analysis; recoveries analysed only among loans labelled Charged Off.' AS scope_note
FROM charged_off;

-- QUERY: by_grade
WITH source AS (
    SELECT *,
        COALESCE(TRY_CAST(recoveries AS DOUBLE), 0.0) AS recovery_amount,
        COALESCE(TRY_CAST(collection_recovery_fee AS DOUBLE), 0.0) AS recovery_fee,
        COALESCE(TRY_CAST(funded_amnt AS DOUBLE), 0.0) AS funded_amount
    FROM loan_source
), charged_off AS (
    SELECT * FROM source WHERE loan_status = 'Charged Off'
)
SELECT
    grade,
    COUNT(*) AS charged_off_loans,
    COUNT(*) FILTER (WHERE recovery_amount > 0) AS loans_with_recovery,
    ROUND(100.0 * COUNT(*) FILTER (WHERE recovery_amount > 0) / NULLIF(COUNT(*), 0), 4) AS recovery_incidence_pct,
    ROUND(SUM(recovery_amount), 2) AS total_gross_recoveries,
    ROUND(AVG(recovery_amount), 2) AS mean_recovery_all_charged_off,
    ROUND(MEDIAN(recovery_amount), 2) AS median_recovery_all_charged_off,
    ROUND(AVG(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS mean_recovery_when_positive,
    ROUND(MEDIAN(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS median_recovery_when_positive,
    ROUND(SUM(recovery_fee), 2) AS total_collection_recovery_fee,
    ROUND(SUM(recovery_amount - recovery_fee), 2) AS net_recovery_after_fee,
    ROUND(100.0 * SUM(recovery_amount) / NULLIF(SUM(funded_amount), 0), 4) AS gross_recovery_pct_of_funded_amount
FROM charged_off
GROUP BY grade
ORDER BY grade;

-- QUERY: by_purpose
WITH source AS (
    SELECT *,
        COALESCE(TRY_CAST(recoveries AS DOUBLE), 0.0) AS recovery_amount,
        COALESCE(TRY_CAST(collection_recovery_fee AS DOUBLE), 0.0) AS recovery_fee,
        COALESCE(TRY_CAST(funded_amnt AS DOUBLE), 0.0) AS funded_amount
    FROM loan_source
), charged_off AS (
    SELECT * FROM source WHERE loan_status = 'Charged Off'
)
SELECT
    purpose,
    COUNT(*) AS charged_off_loans,
    COUNT(*) FILTER (WHERE recovery_amount > 0) AS loans_with_recovery,
    ROUND(100.0 * COUNT(*) FILTER (WHERE recovery_amount > 0) / NULLIF(COUNT(*), 0), 4) AS recovery_incidence_pct,
    ROUND(SUM(recovery_amount), 2) AS total_gross_recoveries,
    ROUND(AVG(recovery_amount), 2) AS mean_recovery_all_charged_off,
    ROUND(MEDIAN(recovery_amount), 2) AS median_recovery_all_charged_off,
    ROUND(AVG(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS mean_recovery_when_positive,
    ROUND(MEDIAN(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS median_recovery_when_positive,
    ROUND(SUM(recovery_fee), 2) AS total_collection_recovery_fee,
    ROUND(SUM(recovery_amount - recovery_fee), 2) AS net_recovery_after_fee,
    ROUND(100.0 * SUM(recovery_amount) / NULLIF(SUM(funded_amount), 0), 4) AS gross_recovery_pct_of_funded_amount
FROM charged_off
GROUP BY purpose
ORDER BY charged_off_loans DESC, purpose;

-- QUERY: by_issue_year
WITH source AS (
    SELECT *,
        COALESCE(TRY_CAST(recoveries AS DOUBLE), 0.0) AS recovery_amount,
        COALESCE(TRY_CAST(collection_recovery_fee AS DOUBLE), 0.0) AS recovery_fee,
        COALESCE(TRY_CAST(funded_amnt AS DOUBLE), 0.0) AS funded_amount,
        TRY_STRPTIME(CAST(issue_d AS VARCHAR), '%b-%y') AS issue_date_parsed
    FROM loan_source
), charged_off AS (
    SELECT *, YEAR(issue_date_parsed) AS issue_year
    FROM source
    WHERE loan_status = 'Charged Off'
)
SELECT
    issue_year,
    COUNT(*) AS charged_off_loans,
    COUNT(*) FILTER (WHERE recovery_amount > 0) AS loans_with_recovery,
    ROUND(100.0 * COUNT(*) FILTER (WHERE recovery_amount > 0) / NULLIF(COUNT(*), 0), 4) AS recovery_incidence_pct,
    ROUND(SUM(recovery_amount), 2) AS total_gross_recoveries,
    ROUND(AVG(recovery_amount), 2) AS mean_recovery_all_charged_off,
    ROUND(MEDIAN(recovery_amount), 2) AS median_recovery_all_charged_off,
    ROUND(AVG(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS mean_recovery_when_positive,
    ROUND(MEDIAN(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS median_recovery_when_positive,
    ROUND(SUM(recovery_fee), 2) AS total_collection_recovery_fee,
    ROUND(SUM(recovery_amount - recovery_fee), 2) AS net_recovery_after_fee,
    ROUND(100.0 * SUM(recovery_amount) / NULLIF(SUM(funded_amount), 0), 4) AS gross_recovery_pct_of_funded_amount
FROM charged_off
WHERE issue_year IS NOT NULL
GROUP BY issue_year
ORDER BY issue_year;

-- QUERY: recovery_amount_bands
WITH source AS (
    SELECT *,
        COALESCE(TRY_CAST(recoveries AS DOUBLE), 0.0) AS recovery_amount,
        COALESCE(TRY_CAST(collection_recovery_fee AS DOUBLE), 0.0) AS recovery_fee,
        COALESCE(TRY_CAST(funded_amnt AS DOUBLE), 0.0) AS funded_amount
    FROM loan_source
), charged_off AS (
    SELECT * FROM source WHERE loan_status = 'Charged Off'
), banded AS (
    SELECT *,
        CASE
            WHEN recovery_amount <= 0 THEN '00. No recovery'
            WHEN recovery_amount <= 100 THEN '01. Greater than 0 to 100'
            WHEN recovery_amount <= 1000 THEN '02. 100 to 1,000'
            WHEN recovery_amount <= 5000 THEN '03. 1,000 to 5,000'
            WHEN recovery_amount <= 10000 THEN '04. 5,000 to 10,000'
            WHEN recovery_amount <= 50000 THEN '05. 10,000 to 50,000'
            ELSE '06. More than 50,000'
        END AS recovery_amount_band
    FROM charged_off
)
SELECT
    recovery_amount_band,
    COUNT(*) AS charged_off_loans,
    COUNT(*) FILTER (WHERE recovery_amount > 0) AS loans_with_recovery,
    ROUND(100.0 * COUNT(*) FILTER (WHERE recovery_amount > 0) / NULLIF(COUNT(*), 0), 4) AS recovery_incidence_pct,
    ROUND(SUM(recovery_amount), 2) AS total_gross_recoveries,
    ROUND(AVG(recovery_amount), 2) AS mean_recovery_all_charged_off,
    ROUND(MEDIAN(recovery_amount), 2) AS median_recovery_all_charged_off,
    ROUND(AVG(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS mean_recovery_when_positive,
    ROUND(MEDIAN(recovery_amount) FILTER (WHERE recovery_amount > 0), 2) AS median_recovery_when_positive,
    ROUND(SUM(recovery_fee), 2) AS total_collection_recovery_fee,
    ROUND(SUM(recovery_amount - recovery_fee), 2) AS net_recovery_after_fee,
    ROUND(100.0 * SUM(recovery_amount) / NULLIF(SUM(funded_amount), 0), 4) AS gross_recovery_pct_of_funded_amount
FROM banded
GROUP BY recovery_amount_band
ORDER BY recovery_amount_band;
