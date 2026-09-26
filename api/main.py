"""
FastAPI app for HR attrition prediction — deployment-ready.
Run: uvicorn api.main:app --reload
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware
from api.predict import get_model, predict_one

app = FastAPI(
    title="HR Attrition Prediction API",
    description="Predict employee attrition from HR features. Model: Random Forest on SQL-engineered features.",
    version="1.0.0",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Request body: one row of features (must match training schema)
class PredictRequest(BaseModel):
    Age: int = Field(..., ge=18, le=70)
    BusinessTravel: int = Field(..., ge=0, le=2)  # 0=Non-Travel, 1=Travel_Rarely, 2=Travel_Frequently
    DailyRate: int = Field(..., ge=100, le=1500)
    Department: str = Field(..., description="e.g. Sales, Research & Development, Human Resources")
    DistanceFromHome: int = Field(..., ge=1, le=30)
    Education: int = Field(..., ge=1, le=5)
    EducationField: str = Field(..., description="e.g. Life Sciences, Medical, Other")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str = Field(..., description="Male or Female")
    HourlyRate: int = Field(..., ge=30, le=100)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str = Field(..., description="e.g. Sales Executive, Research Scientist")
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: str = Field(..., description="Single, Married, Divorced")
    MonthlyIncome: int = Field(..., ge=1000, le=20000)
    MonthlyRate: int = Field(..., ge=2000, le=27000)
    NumCompaniesWorked: int = Field(..., ge=0, le=10)
    OverTime: int = Field(..., ge=0, le=1)
    PercentSalaryHike: int = Field(..., ge=11, le=25)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=45)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=45)
    YearsInCurrentRole: int = Field(..., ge=0, le=20)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=20)
    YearsWithCurrManager: int = Field(..., ge=0, le=20)


@app.get("/health")
def health():
    """Liveness/readiness for Docker and cloud platforms."""
    return {"status": "ok", "service": "hr-attrition-api"}


@app.get("/")
def root():
    return {
        "message": "HR Attrition Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict attrition for one employee. Returns prediction (0/1) and probabilities."""
    try:
        features = req.model_dump()
        result = predict_one(features)
        return result
    except FileNotFoundError as e:
        # 503 is appropriate for "Service Unavailable" if the model isn't loaded
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Please ensure it is trained and loaded.")
    except Exception as e:
        # Log the actual error to avoid information disclosure
        import logging
        logging.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="An error occurred while processing the prediction request. Ensure all features are provided correctly.")


@app.on_event("startup")
def startup():
    """Load model at startup so first request is fast."""
    try:
        get_model()
    except FileNotFoundError:
        pass  # Will fail on first /predict with clear error
