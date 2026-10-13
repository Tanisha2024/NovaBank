
# NovaBank — Temporal Validation

Rows with valid dates: 38577
Issuance period: 2007-06-01 to 2011-12-01

Temporal cutoff: 2011-02
Training rows: 19441
Test rows: 19136
Training dates: 2007-06-01 to 2011-01-01
Test dates: 2011-02-01 to 2011-12-01
Training charged-off rate: 13.15 %
Test charged-off rate: 16.04 %

Temporal model training complete.

===== TEMPORAL VALIDATION RESULTS =====
training_rows: 19441
test_rows: 19136
training_bad_rate_pct: 13.15
test_bad_rate_pct: 16.04
roc_auc: 0.6674
average_precision: 0.2701
test_prevalence_baseline: 0.1604
brier_score_model: 0.1291
brier_score_constant_baseline: 0.1355

Metrics and test predictions saved.
## Limitations
The evaluation uses resolved loans from a historical US lending dataset.
Temporal separation reduces one form of leakage but does not eliminate
selection bias or guarantee performance on future loan applicants.
# NovaBank — Temporal Validation

## Method

* Model: Logistic Regression
* Training period: June 2007–January 2011
* Test period: February–December 2011
* Split unit: Calendar month
* Training records: 19,441
* Test records: 19,136

## Results

* Training charged-off rate: 13.15%
* Test charged-off rate: 16.04%
* ROC-AUC: 0.6674
* Average precision: 0.2701
* Test prevalence baseline for average precision: 0.1604
* Model Brier score: 0.1291
* Constant-probability baseline Brier score: 0.1355

## Interpretation

The model showed moderate ranking performance on the later-issued temporal test set. Its average precision exceeded the test-set prevalence baseline, and its Brier score was lower than that of a constant-probability predictor.

The test period had a higher charged-off rate than the training period, indicating a difference in observed outcome prevalence. This warrants further investigation.

These results do not establish that the model will generalize to all future borrowers.

## Limitations

* Historical US lending data from 2007–2011.
* Only loans with observed Fully Paid or Charged Off outcomes were included.
* Excluding Current loans creates potential outcome-selection bias, especially in later issuance months.
* Temporal separation does not eliminate selection bias or guarantee future performance.
* Predictive ranking is not evidence of financial losses prevented.
* Probability calibration and review-capacity results require further assessment.

## Next Steps

1. Evaluate model calibration on the temporal test set.
2. Compare review-capacity results at 5%, 10%, 15% and 20%.
3. Compare Logistic Regression against a stronger model using the same temporal evaluation design.
4. Assess performance across relevant borrower segments.
5. Document the limits of generalizing historical US lending results to modern lending decisions.
