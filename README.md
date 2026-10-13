# NovaBank — AI-Assisted Financial Risk Intelligence

NovaBank is a portfolio project combining credit-risk analytics, synthetic transaction fraud detection, collections and recovery analysis, explainable machine learning, a Groq-powered risk explanation feature, SQL workflows, and a Streamlit application.

## Project Modules

### 1. Credit Risk Analytics
- Analyze historical LendingClub loan outcomes and charge-off trends.
- Explore risk across loan grades, purposes, income, and debt-to-income ratios.
- Evaluate a temporal XGBoost model and investigate risk drivers.
- Examine individual historical loans using the Risk Explorer when the required local data is available.

Reported analysis: 38,577 loans, 5,627 charged-off loans, and a 14.59% charge-off rate.

Reported temporal test metrics: ROC-AUC 0.6674 and average precision 0.2701.

### 2. Fraud Detection
- Interactive fraud-scoring playground using saved XGBoost models.
- Feature-set comparisons and feature-importance analysis.
- SHAP-based explanations and threshold evaluation.

Reported no-step model test metrics:
- ROC-AUC: 0.9021
- Average precision: 0.5487
- Precision: 0.5876
- Recall: 0.4633
- F1: 0.5181

These metrics come from a synthetic PaySim dataset and the specified test split. They do not establish real-world bank performance. The model score is not a calibrated real-world fraud probability.

### 3. Collections & Recovery
- Analyze recovery incidence among charged-off loans.
- Compare recovery patterns by grade, issue year, purpose, and amount band.
- Examine gross recoveries, recorded recovery fees, and net recovery after those fees.

Reported historical analysis: 5,627 charged-off loans, 4,218 with positive recovery, $3,781,917.24 in gross recoveries, $492,733.55 in recorded recovery fees, and $3,289,183.69 net after those fees.

### 4. Power BI
A local Power BI report was created with Credit Risk, Fraud Detection, and Collections & Recovery pages. The `.pbix` file is not included in this repository while its embedded data and redistribution permissions remain under review.

## Technology Stack
- Python, pandas, NumPy, SciPy
- scikit-learn, XGBoost, SHAP
- Streamlit
- Groq API
- DuckDB and SQL
- Jupyter notebooks
- Power BI

## Repository Structure

```text
NovaBank/
├── app/             # Streamlit app and fraud playground
├── models/          # Saved model artifacts
├── notebooks/       # Data audit, EDA, modeling, explainability
├── reports/         # Aggregate analysis outputs and metrics
├── sql/             # SQL analysis and validation
├── src/             # Reusable Python modules
├── business_case.md
├── requirements.txt
└── README.md
```

## Setup

Use a Python environment compatible with the pinned dependencies in `requirements.txt`.

### 1. Clone the repository

```bash
git clone https://github.com/Tanisha2024/NovaBank.git
cd NovaBank
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the Groq API key

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder with your own key:

```text
GROQ_API_KEY=your_actual_key
```

Never commit `.env` or publish API keys.

### 4. Prepare local data

Raw datasets and generated row-level datasets are excluded from this repository. The credit-risk cohort expected by the Risk Explorer must be prepared locally at:

`data/processed/credit_risk_cohort.csv`

Some workflows require source datasets to be obtained separately and placed at their expected local paths. Follow applicable source terms and permissions. Not every application page will work until its required local data is available.

### 5. Start the Streamlit app

From the repository root:

```powershell
python -m streamlit run app/app.py
```

Open the local URL printed in the terminal. The AI explanation feature requires a valid Groq API key and network access.

## Data, Evaluation, and Limitations

- LendingClub results describe historical loan data and may not generalize to current lending portfolios.
- PaySim is synthetic transaction data; fraud metrics should not be interpreted as real-bank performance.
- Reported metrics apply to their documented evaluation splits and thresholds.
- Model associations are not causal evidence, and AI-generated explanations can be inaccurate.
- Large raw datasets, row-level prediction exports, and the local Power BI file are intentionally excluded.
- Reproducing all analyses or retraining the models requires the relevant source data.

This project is intended for educational and portfolio demonstration purposes. It is not a validated production banking or automated financial decision system.
