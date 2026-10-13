-- QUERY: baseline

SELECT
    COUNT(*) AS resolved_loans,

    SUM(CASE
        WHEN bad_outcome = 1 THEN 1 ELSE 0
    END) AS charged_off_loans,

    SUM(CASE
        WHEN bad_outcome = 0 THEN 1 ELSE 0
    END) AS fully_paid_loans,

    ROUND(
        100.0 * SUM(
            CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS charged_off_rate_pct

FROM read_csv_auto(
    'data/processed/credit_risk_cohort.csv',
    header = true
);
-- QUERY: grade

WITH credit AS (
    SELECT *
    FROM read_csv_auto(
        'data/processed/credit_risk_cohort.csv',
        header = true
    )
),

portfolio AS (
    SELECT
        SUM(
            CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
        ) AS total_bad_loans
    FROM credit
)

SELECT
    grade,
    COUNT(*) AS loan_count,

    SUM(
        CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
    ) AS bad_loans,

    ROUND(
        100.0 * SUM(
            CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS bad_rate_pct,

    ROUND(
        100.0 * SUM(
            CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
        ) / (SELECT total_bad_loans FROM portfolio),
        2
    ) AS bad_loan_share_pct

FROM credit
WHERE grade IS NOT NULL
GROUP BY grade
ORDER BY bad_rate_pct DESC;

-- QUERY: dti_bands

WITH credit AS (
    SELECT *
    FROM read_csv_auto(
        'data/processed/credit_risk_cohort.csv',
        header = true
    )
),

banded AS (
    SELECT
        CASE
            WHEN dti <= 6.91
                THEN '(-0.001, 6.91]'
            WHEN dti <= 11.35
                THEN '(6.91, 11.35]'
            WHEN dti <= 15.31
                THEN '(11.35, 15.31]'
            WHEN dti <= 19.71
                THEN '(15.31, 19.71]'
            ELSE '(19.71, 29.99]'
        END AS band,

        bad_outcome

    FROM credit
    WHERE dti IS NOT NULL
)

SELECT
    band,
    COUNT(*) AS loan_count,

    SUM(
        CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
    ) AS bad_loans,

    ROUND(
        100.0 * SUM(
            CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS bad_rate_pct

FROM banded

GROUP BY band

ORDER BY
    CASE band
        WHEN '(-0.001, 6.91]' THEN 1
        WHEN '(6.91, 11.35]' THEN 2
        WHEN '(11.35, 15.31]' THEN 3
        WHEN '(15.31, 19.71]' THEN 4
        ELSE 5
    END;

-- QUERY: income_bands

WITH credit AS (
    SELECT *
    FROM read_csv_auto(
        'data/processed/credit_risk_cohort.csv',
        header = true
    )
),

banded AS (
    SELECT
        CASE
            WHEN annual_inc <= 37196.16
                THEN '(3999.999, 37196.16]'
            WHEN annual_inc <= 50004.0
                THEN '(37196.16, 50004.0]'
            WHEN annual_inc <= 65004.0
                THEN '(50004.0, 65004.0]'
            WHEN annual_inc <= 90000.0
                THEN '(65004.0, 90000.0]'
            ELSE '(90000.0, 6000000.0]'
        END AS band,

        bad_outcome

    FROM credit
    WHERE annual_inc IS NOT NULL
)

SELECT
    band,
    COUNT(*) AS loan_count,

    SUM(
        CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
    ) AS bad_loans,

    ROUND(
        100.0 * SUM(
            CASE WHEN bad_outcome = 1 THEN 1 ELSE 0 END
        ) / COUNT(*),
        2
    ) AS bad_rate_pct

FROM banded

GROUP BY band

ORDER BY
    CASE band
        WHEN '(3999.999, 37196.16]' THEN 1
        WHEN '(37196.16, 50004.0]' THEN 2
        WHEN '(50004.0, 65004.0]' THEN 3
        WHEN '(65004.0, 90000.0]' THEN 4
        ELSE 5
    END;