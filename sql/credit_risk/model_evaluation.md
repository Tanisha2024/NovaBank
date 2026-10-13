# NovaBank — Credit Risk Model Evaluation

## 1. Model

* Model: Logistic Regression
* Outcome: `bad_outcome`
* Positive class: Charged Off
* Evaluation approach: Stratified random train-test split
* Training proportion: 80%
* Test proportion: 20%
* Random seed: 42
* Test records: 7,716

## 2. Evaluation Results

| Metric                          | Result |
| ------------------------------- | -----: |
| Test charged-off rate           | 14.58% |
| ROC-AUC                         | 0.6620 |
| Average precision               | 0.2556 |
| Brier score — model             | 0.1189 |
| Brier score — constant baseline | 0.1245 |

The model demonstrates moderate ranking ability and performs better than the constant-probability baseline on average precision and Brier score. Its probability calibration and future-period performance have not yet been validated.

## 3. Review-Capacity Experiment

At a review capacity of 10% of the test set:

* Loans reviewed: 772
* Charged-off loans in the test set: 1,125
* Charged-off loans ranked in the reviewed group: 226
* Capture rate: 20.09%
* Precision in the reviewed group: 29.27%

The top-ranked 10% contains approximately twice the overall test-set proportion of charged-off loans. However, 20.09% of charged-off loans are captured, leaving most outside the review group.

This result measures retrospective ranking performance, not actual money recovered or losses prevented.

## 4. Model Interpretation

The fitted model has positive coefficients for DTI, revolving-credit utilization and previous delinquencies, while annual income has a negative coefficient.

These coefficients describe conditional associations in the fitted model. They do not establish causal effects or prove that a feature is independently significant.

Missingness indicators also have substantial coefficients. Their contribution should be investigated to determine whether it reflects genuine information, historical data-collection practices or other dataset characteristics.

## 5. Limitations

* Historical US lending data from 2007–2011.
* Current loans excluded from the resolved-outcome sample.
* Random-split evaluation; chronological validation remains necessary.
* No claim of real-world loss reduction or underwriting improvement.
* Calibration, threshold selection and segment-level performance require further evaluation.

## 6. Next Steps

1. Test ranking performance at several review-capacity levels.
2. Evaluate temporal generalization using older loans for training and later loans for testing.
3. Assess probability calibration.
4. Compare Logistic Regression with a stronger model.
5. Evaluate segment-level performance and threshold trade-offs.
