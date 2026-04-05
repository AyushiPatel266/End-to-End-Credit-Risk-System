from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load artifacts
model         = joblib.load('models/model.pkl')
scaler        = joblib.load('models/scaler.pkl')
threshold     = joblib.load('models/threshold.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# App
app = FastAPI(
    title="Credit Risk Prediction API",
    description="""
    End-to-End Credit Risk Prediction System built with XGBoost.
    
    This API predicts the probability that a loan applicant will experience 
    financial distress within the next 2 years.
    
    Trained on 150,000 real loan records from the Give Me Some Credit dataset (Kaggle).
    """,
    version="1.0.0"
)

# Request schema
class ApplicantData(BaseModel):
    revolving_utilization: float = Field(
        ..., ge=0.0, le=1.0,
        description="Ratio of revolving credit used vs total limit (0 to 1)",
        example=0.3
    )
    age: int = Field(
        ..., ge=18, le=100,
        description="Applicant age in years",
        example=35
    )
    times_30_59_days_late: int = Field(
        ..., ge=0, le=20,
        description="Number of times 30-59 days past due",
        example=0
    )
    debt_ratio: float = Field(
        ..., ge=0.0, le=10.0,
        description="Monthly debt payments divided by monthly income",
        example=0.35
    )
    monthly_income: float = Field(
        ..., ge=0,
        description="Monthly income in USD",
        example=5000
    )
    number_of_open_credit_lines: int = Field(
        ..., ge=0, le=30,
        description="Number of open credit lines and loans",
        example=5
    )
    times_90_days_late: int = Field(
        ..., ge=0, le=20,
        description="Number of times 90+ days past due",
        example=0
    )
    number_real_estate_loans: int = Field(
        ..., ge=0, le=10,
        description="Number of mortgage and real estate loans",
        example=1
    )
    times_60_89_days_late: int = Field(
        ..., ge=0, le=20,
        description="Number of times 60-89 days past due",
        example=0
    )
    number_of_dependents: int = Field(
        ..., ge=0, le=10,
        description="Number of dependents in family",
        example=1
    )

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    default_probability: float
    decision:            str
    risk_level:          str
    threshold_used:      float
    recommendation:      str
    input_summary:       dict

# Routes
@app.get("/")
def root():
    return {
        "message": "Credit Risk Prediction API is running",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model":         "XGBoost",
        "features":      len(feature_names),
        "threshold":     round(float(threshold), 4)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ApplicantData):
    try:
        # Build input dataframe in correct feature order
        input_data = pd.DataFrame([[
            data.revolving_utilization,
            data.age,
            data.times_30_59_days_late,
            data.debt_ratio,
            data.monthly_income,
            data.number_of_open_credit_lines,
            data.times_90_days_late,
            data.number_real_estate_loans,
            data.times_60_89_days_late,
            data.number_of_dependents
        ]], columns=feature_names)

        # Predict
        prob       = float(model.predict_proba(input_data)[0][1])
        is_high    = prob >= float(threshold)
        decision   = "REJECT" if is_high else "APPROVE"

        # Risk level
        if prob > 0.7:
            risk_level = "Critical"
        elif prob > float(threshold):
            risk_level = "High"
        elif prob > 0.3:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # Recommendation
        if is_high:
            recommendation = (
                f"Applicant has a {prob*100:.1f}% probability of default, "
                f"exceeding the {float(threshold)*100:.1f}% threshold. "
                f"Recommend rejection or request for additional collateral."
            )
        else:
            recommendation = (
                f"Applicant has only a {prob*100:.1f}% probability of default, "
                f"below the {float(threshold)*100:.1f}% threshold. "
                f"Recommend approval with standard loan terms."
            )

        return PredictionResponse(
            default_probability = round(prob, 4),
            decision            = decision,
            risk_level          = risk_level,
            threshold_used      = round(float(threshold), 4),
            recommendation      = recommendation,
            input_summary       = data.dict()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(applicants: list[ApplicantData]):
    """
    Predict for multiple applicants at once.
    Send a list of applicant objects and get back predictions for all.
    """
    try:
        results = []
        for applicant in applicants:
            input_data = pd.DataFrame([[
                applicant.revolving_utilization,
                applicant.age,
                applicant.times_30_59_days_late,
                applicant.debt_ratio,
                applicant.monthly_income,
                applicant.number_of_open_credit_lines,
                applicant.times_90_days_late,
                applicant.number_real_estate_loans,
                applicant.times_60_89_days_late,
                applicant.number_of_dependents
            ]], columns=feature_names)

            prob     = float(model.predict_proba(input_data)[0][1])
            is_high  = prob >= float(threshold)
            decision = "REJECT" if is_high else "APPROVE"

            results.append({
                "default_probability": round(prob, 4),
                "decision":            decision,
                "risk_level":          "Critical" if prob > 0.7 else
                                       "High"     if prob > float(threshold) else
                                       "Moderate" if prob > 0.3 else "Low",
                "input":               applicant.dict()
            })

        return {
            "total_applicants": len(results),
            "approved":         sum(1 for r in results if r["decision"] == "APPROVE"),
            "rejected":         sum(1 for r in results if r["decision"] == "REJECT"),
            "results":          results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))