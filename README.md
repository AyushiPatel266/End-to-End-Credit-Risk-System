# End-to-End Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready machine learning system that predicts the probability of a loan applicant defaulting within 2 years. Built with a focus on real-world ML practices, class imbalance handling, experiment tracking, model explainability, and live deployment.

**Live Demo:** [end-to-end-credit-risk-system.streamlit.app](https://end-to-end-credit-risk-system.streamlit.app)

---

## What this project does

Banks lose billions approving loans to high-risk applicants. At the same time, rejecting good customers means lost revenue. This system helps make that decision smarter using machine learning.

Given an applicant's financial profile, the model returns:
- Default probability score
- Approve / Reject decision
- Risk level (Low / Moderate / High / Critical)
- SHAP explanation of exactly which factors drove the decision

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11 |
| ML Model | XGBoost |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| Frontend | Streamlit |
| API | FastAPI |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

## Dataset

**Give Me Some Credit** - Kaggle Competition Dataset

- 150,000 real loan applicant records
- 10 financial features
- Binary target: financial distress within 2 years
- Source: [kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)

> Dataset not included in this repo due to size. Download from Kaggle and place in `data/` folder.

---

## Project Structure
End-to-End-Credit-Risk-System/
├── data/
│   ├── Data Dictionary.xls
│   └── processed/
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Model_Evaluation.ipynb
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── threshold.pkl
│   └── feature_names.pkl
├── app.py
├── api.py
├── requirements.txt
└── README.md

---

## ML Pipeline

### 1. Exploratory Data Analysis
- Analyzed distributions, missing values, and outliers
- Visualized default rate by age group and feature correlations
- Identified class imbalance: only 6.7% defaulted

### 2. Preprocessing
- Imputed missing values using median (robust to outliers)
- Capped outliers at business-logical thresholds
- Stratified 80/20 train/test split

### 3. Model Training
- Trained 4 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Applied SMOTE to fix class imbalance on training data only
- Tracked all experiments with MLflow for full reproducibility
- Tuned XGBoost with RandomizedSearchCV across 20 parameter combinations

### 4. Evaluation
- Evaluated using ROC-AUC, Precision, Recall, and Confusion Matrix
- Tuned classification threshold for optimal F1 score
- Analyzed business cost of false positives vs false negatives

### 5. Explainability
- Used SHAP TreeExplainer to explain individual predictions
- Each prediction shows which features pushed the risk score up or down
- Critical for regulatory compliance in real banking environments

### 6. Deployment
- Streamlit frontend deployed on Streamlit Cloud (live link above)
- FastAPI REST API with `/predict` and `/predict/batch` endpoints
- Supports both single and batch predictions

---

## Business Metrics

| Metric | What it means in business terms |
|--------|--------------------------------|
| False Negative | Approved a bad loan - bank absorbs the loss |
| False Positive | Rejected a good customer - lost revenue |
| Threshold tuning | Optimized the decision boundary for business cost |
| SHAP | Explains each decision - required by financial regulators |

---

## How to Run Locally
```bash
# Clone the repo
git clone https://github.com/AyushiPatel266/End-to-End-Credit-Risk-System.git
cd End-to-End-Credit-Risk-System

# Create environment
conda create -n creditrisk python=3.11
conda activate creditrisk

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place cs-training.csv in data/ folder as credit_data.csv

# Run notebooks in order
# 01 → 02 → 03

# Launch Streamlit app
streamlit run app.py

# Launch FastAPI (optional)
uvicorn api:app --reload
```

---

## API Usage
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "revolving_utilization": 0.3,
    "age": 35,
    "times_30_59_days_late": 0,
    "debt_ratio": 0.35,
    "monthly_income": 5000,
    "number_of_open_credit_lines": 5,
    "times_90_days_late": 0,
    "number_real_estate_loans": 1,
    "times_60_89_days_late": 0,
    "number_of_dependents": 1
  }'
```

---

## Key Learnings

- Class imbalance is one of the most common real-world ML challenges. SMOTE helped balance the training data without touching the test set
- Accuracy is a misleading metric for imbalanced datasets; ROC-AUC and threshold tuning matter more
- A deployed model with explainability is worth more than a perfect model in a notebook
- Regulators require loan decisions to be explainable, SHAP provides that at the individual prediction level

---

## Author

**Ayushi Patel**
- GitHub: [AyushiPatel266](https://github.com/AyushiPatel266)
- Live App: [end-to-end-credit-risk-system.streamlit.app](https://end-to-end-credit-risk-system.streamlit.app)
