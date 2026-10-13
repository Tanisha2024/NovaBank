-- NovaBank | Fraud Detection Intelligence
-- Run scripts/run_fraud_sql.py from the NovaBank project root.
-- DuckDB reads the original CSV directly; do not open/resave it in Excel.

CREATE OR REPLACE VIEW paysim AS
SELECT *
FROM read_csv_auto('data/raw/paysim.csv', header = true);

-- QUERY: baseline_metrics
SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN isFraud = 0 THEN 1 ELSE 0 END) AS normal_transactions,
    ROUND(100.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS total_amount_on_fraud_labeled_transactions
FROM paysim;

-- QUERY: fraud_by_transaction_type
SELECT
    type AS transaction_type,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 5) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS fraud_labeled_transaction_amount
FROM paysim
GROUP BY type
ORDER BY fraud_rate_pct DESC, total_transactions DESC;

-- QUERY: fraud_by_amount_band
SELECT
    CASE
        WHEN amount < 1000 THEN '01. Below 1,000'
        WHEN amount < 10000 THEN '02. 1,000 to 9,999'
        WHEN amount < 100000 THEN '03. 10,000 to 99,999'
        WHEN amount < 1000000 THEN '04. 100,000 to 999,999'
        ELSE '05. 1,000,000 or more'
    END AS amount_band,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 5) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS average_transaction_amount
FROM paysim
GROUP BY amount_band
ORDER BY amount_band;

-- QUERY: fraud_by_step
SELECT
    step,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 5) AS fraud_rate_pct
FROM paysim
GROUP BY step
ORDER BY step;

-- QUERY: flag_vs_fraud_label
SELECT
    isFlaggedFraud,
    isFraud,
    COUNT(*) AS transaction_count
FROM paysim
GROUP BY isFlaggedFraud, isFraud
ORDER BY isFlaggedFraud, isFraud;

-- QUERY: balance_mismatch_by_label
SELECT
    isFraud,
    COUNT(*) AS transaction_count,
    ROUND(AVG(ABS((oldbalanceOrg - amount) - newbalanceOrig)), 2) AS avg_origin_balance_mismatch,
    ROUND(quantile_cont(ABS((oldbalanceOrg - amount) - newbalanceOrig), 0.5), 2) AS median_origin_balance_mismatch,
    ROUND(AVG(ABS((oldbalanceDest + amount) - newbalanceDest)), 2) AS avg_destination_balance_mismatch,
    ROUND(quantile_cont(ABS((oldbalanceDest + amount) - newbalanceDest), 0.5), 2) AS median_destination_balance_mismatch
FROM paysim
GROUP BY isFraud
ORDER BY isFraud;
