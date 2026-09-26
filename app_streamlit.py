"""
Streamlit demo: HR Attrition Predictor.
Run: streamlit run app_streamlit.py
Deploy: Streamlit Community Cloud (connect GitHub repo).
"""
from pathlib import Path

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="HR Attrition Predictor", page_icon="📊", layout="centered")

MODEL_PATH = Path(__file__).parent / "output" / "attrition_model.pkl"

st.title("📊 HR Attrition Predictor")
st.markdown("Predict whether an employee is at risk of leaving. Built with SQL-engineered features and a Random Forest model.")

if not MODEL_PATH.exists():
    st.error(f"Model not found. Run the pipeline first: `python run_pipeline.py`")
    st.stop()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipe = load_model()
preprocess = pipe.named_steps.get("preprocess")
feature_names = list(preprocess.feature_names_in_) if preprocess is not None and hasattr(preprocess, "feature_names_in_") else []

with st.form("employee_form"):
    st.subheader("Employee profile")
    c1, c2 = st.columns(2)
    with c1:
        Age = st.slider("Age", 18, 65, 35)
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        JobRole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        OverTime = st.selectbox("Over Time", ["No", "Yes"])
    with c2:
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
        YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
        YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 2)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        JobSatisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        EnvironmentSatisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        WorkLifeBalance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        DistanceFromHome = st.slider("Distance from Home (miles)", 1, 30, 10)

    # Fill remaining required fields with typical defaults
    DailyRate = MonthlyIncome // 30
    MonthlyRate = MonthlyIncome * 2
    HourlyRate = MonthlyIncome // 160
    Education = 3
    JobInvolvement = 3
    JobLevel = 2
    NumCompaniesWorked = 2
    PercentSalaryHike = 14
    PerformanceRating = 3
    RelationshipSatisfaction = 3
    StockOptionLevel = 1
    TotalWorkingYears = YearsAtCompany + 2
    TrainingTimesLastYear = 2
    YearsWithCurrManager = min(YearsInCurrentRole + 1, 10)

    bt_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "Age": Age, "BusinessTravel": bt_map[BusinessTravel], "DailyRate": DailyRate,
        "Department": Department, "DistanceFromHome": DistanceFromHome, "Education": Education,
        "EducationField": EducationField, "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender, "HourlyRate": HourlyRate, "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel, "JobRole": JobRole, "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus, "MonthlyIncome": MonthlyIncome, "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked, "OverTime": 1 if OverTime == "Yes" else 0,
        "PercentSalaryHike": PercentSalaryHike, "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction, "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears, "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance, "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole, "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
    }
    df = pd.DataFrame([row]).reindex(columns=feature_names)
    proba = pipe.predict_proba(df)[0]
    pred = int(pipe.predict(df)[0])

    st.success("Prediction ready")
    risk_pct = round(proba[1] * 100, 1)
    if pred == 1:
        st.error(f"**At risk of leaving** — {risk_pct}% probability of attrition. Consider retention actions.")
    else:
        st.info(f"**Low risk** — {risk_pct}% probability of attrition.")
    st.caption(f"Probability Stay: {proba[0]:.2%}  |  Probability Leave: {proba[1]:.2%}")

st.markdown("---")
st.caption("HR Analytics ML Pipeline — SQL + Random Forest | [GitHub](https://github.com)")
