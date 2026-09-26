"""
ML training and evaluation for attrition prediction.
Uses features produced by SQL (or fallback from raw table).
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES_IBM = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
CATEGORICAL_FEATURES_IBM = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]
TARGET = "target"


def _get_numeric_categorical(df: pd.DataFrame):
    """Determine numeric vs categorical columns for modeling (exclude target)."""
    numeric = []
    categorical = []
    for c in df.columns:
        if c == TARGET:
            continue
        if c in NUMERIC_FEATURES_IBM or (df[c].dtype in ("int64", "float64") and df[c].nunique() > 10):
            numeric.append(c)
        elif c in CATEGORICAL_FEATURES_IBM or df[c].dtype == "object" or df[c].nunique() < 20:
            categorical.append(c)
    return numeric, categorical


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    model_path: str | Path | None = None,
) -> dict:
    """
    Train a Random Forest on the ML feature dataframe (target column must be 'target').
    Returns metrics dict and optionally saves the fitted pipeline.
    """
    if TARGET not in df.columns:
        raise ValueError(f"DataFrame must contain a '{TARGET}' column (binary attrition).")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols, categorical_cols = _get_numeric_categorical(df)
    # Ensure we only use columns present in X
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
    pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_weighted")

    metrics = {
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "cv_f1_weighted_mean": float(cv_scores.mean()),
        "cv_f1_weighted_std": float(cv_scores.std()),
    }

    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)

    return metrics, pipe
