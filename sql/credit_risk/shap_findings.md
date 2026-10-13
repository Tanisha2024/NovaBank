# NovaBank — Global SHAP Explainability Findings

## Method

* Model: XGBoost binary classifier
* Evaluation population: Later-issued temporal test set
* Explained sample: 500 observations
* Explanation method: SHAP TreeExplainer
* Output being explained: Predicted probability of the positive class
* Feature aggregation: Transformed contributions grouped by original feature

## Top Features

1. Revolving utilization — 0.031949
2. Annual income — 0.028250
3. Loan amount — 0.019323
4. Recent credit inquiries — 0.016268
5. Loan purpose — 0.009432
6. Revolving balance — 0.008492
7. Months since last delinquency — 0.007275
8. Employment length — 0.005932
9. Credit history length — 0.005198
10. Public records — 0.005048

## Interpretation

Revolving utilization has the largest mean absolute grouped SHAP contribution among the features in the explained sample, followed by annual income and loan amount.

These results suggest that credit utilization, financial capacity, borrowing exposure and credit-seeking behaviour contribute substantially to the fitted model's risk predictions.

The importance values describe average contribution magnitudes. They do not establish the direction of an effect for every individual, causal relationships, or the independent predictive value of each variable.

## Limitations

* Results are based on a sample of 500 later-issued historical loans.
* The model was trained on historical US lending data.
* Correlated features may share predictive information.
* Feature importance is not evidence of causation.
* Model explanations do not independently justify lending decisions.

## Next Step

Examine individual-loan explanations to establish which features pushed a particular prediction higher or lower. Use verified model outputs and SHAP contributions as evidence for the AI Risk Explanation Copilot.
