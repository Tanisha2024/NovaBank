# NovaBank Credit Risk — Initial Findings

## 1. Scope

* **Dataset:** LendingClub 2007–2011
* **Population:** Loans with observed `Fully Paid` or `Charged Off` outcomes
* **Target:** `bad_outcome = 1` for `Charged Off`; `0` for `Fully Paid`
* **Analysis type:** Retrospective descriptive analysis

## 2. Portfolio Baseline

* **Number of resolved loans:** 38,577
* **Number of charged-off loans:** 5,627
* **Number of fully paid loans:** 32,950
* **Historical charged-off rate:** 14.59%

The analysis includes loans with observed final outcomes. Current loans are excluded because their ultimate repayment outcomes are not observed in this dataset.

The 14.59% rate describes this historical resolved-loan sample. It should not be interpreted as the default rate of a modern bank or of all loan applicants.

## 3. Segment Analysis

### Highest-Risk Observed Credit Grade

* **Credit grade:** G
* **Segment loan count:** 299
* **Charged-off loans:** 101
* **Charged-off rate:** 33.78%
* **Share of all charged-off loans:** 1.79%
* **95% confidence interval:** 28.65%–39.32%

### Interpretation

Grade G has the highest observed charged-off rate among the credit grades in the resolved-loan sample. Its rate is approximately 2.32 times the overall historical charged-off rate of 14.59%.

However, Grade G contains only 299 loans and accounts for 1.79% of all charged-off loans. The high rate does not necessarily mean this segment contributes the greatest financial loss or represents the largest portfolio-wide opportunity for intervention.

Further analysis should compare all credit grades by charged-off rate, loan count, original loan amount and share of charged-off loans.

These results are descriptive and do not establish that credit grade itself causes a loan to be charged off.

## 4. Financial Characteristics

### 4.1 Debt-to-Income Ratio (DTI)

The historical charged-off rate generally increased across the five DTI bands, from 12.24% in the lowest band to approximately 16.89% in the highest band.

The highest-DTI band contained 7,699 loans, including 1,300 charged-off loans.

This pattern suggests that a higher debt burden relative to income is associated with worse historical loan outcomes in this sample. The relationship requires further investigation alongside income, credit history, loan grade and other borrower characteristics.

### 4.2 Annual Income

The charged-off rate decreased across the five annual-income bands, from 18.31% in the lowest band to 10.93% in the highest band.

The lowest-income band contained 7,716 loans, including 1,413 charged-off loans.

This is an inverse association between annual income and charged-off rate in the observed sample. Income alone does not establish repayment capacity; other factors, including existing debt, loan size and credit history, may contribute to the pattern.

The highest-income band extends to $6,000,000 in the source data. These extreme values should be investigated during data-quality analysis before interpreting the top income band too confidently.

### 4.3 Revolving-Credit Utilization

The charged-off rate increased across the five utilization bands, from 9.94% in the lowest band to 19.81% in the highest band.

The highest-utilization band contained 7,664 loans, including 1,518 charged-off loans.

The observed rate was approximately twice as high in the highest-utilization band as in the lowest band. This suggests a potentially important association between revolving-credit utilization and historical loan outcomes.

Fifty resolved loans were excluded from the utilization-band analysis because their utilization values were missing. The results therefore represent loans with observed utilization values.

### 4.4 Unexpected Findings

The three variables show different but interpretable patterns:

* Higher DTI corresponds to a higher observed charged-off rate.
* Higher revolving-credit utilization corresponds to a higher observed charged-off rate.
* Higher annual income corresponds to a lower observed charged-off rate.

These patterns are descriptive. They do not establish independent effects, causal relationships or reliable predictions for new borrowers. Further analysis is needed to assess overlapping risk factors.

## 5. Business Implications

### 5.1 Segments for Further Investigation

**Grade G:** The observed charged-off rate was 33.78% across 299 loans, compared with the portfolio baseline of 14.59%. The segment represents 1.79% of all charged-off loans. Its financial contribution must be assessed separately.

**High revolving-credit utilization:** The highest band had a 19.81% charged-off rate across 7,664 loans. Both its rate and segment size make it worth investigating.

**High DTI:** The highest band had an approximately 16.89% charged-off rate across 7,699 loans.

**Lower annual-income bands:** The lowest band had an 18.31% charged-off rate across 7,716 loans. The top income band's extreme values should be investigated before drawing strong comparisons.

### 5.2 Additional Information Needed

Further investigation should consider:

* Credit grade, loan amount, term and interest rate.
* Previous delinquencies and employment history.
* The overlap between DTI, annual income and revolving-credit utilization.
* Original loan amounts, actual losses and historical recoveries as separate financial measures.

Original loan amount is not the same as current outstanding balance, and historical charged-off counts do not directly measure financial loss.

### 5.3 Intervention to Test

Develop and evaluate an application-stage credit-risk model using eligible borrower information available at the defined prediction point.

Compare its risk ranking with a simple baseline using a separate validation sample.

Evaluate how alternative decision thresholds affect:

* The number of loans classified as high-risk.
* The proportion of historical charged-off loans identified.
* The trade-off between identifying risky loans and incorrectly flagging loans that were fully paid.

Any proposed underwriting intervention should be evaluated on held-out historical data and reviewed for performance differences across relevant segments before being presented as a recommendation.

## 6. Limitations

* The dataset contains historical US lending records, not a current Indian loan portfolio.
* Current loans are excluded from the initial resolved-outcome analysis.
* Associations do not establish causation.
* Original loan amount is not current outstanding balance.
* The findings reflect a selected resolved-loan sample and historical lending policies.
* Observed charged-off rates do not automatically translate into future default probabilities.
* The initial analysis is descriptive; predictive performance and intervention effectiveness have not yet been established.
