"""
Prediction API for HR attrition model.
Loads the trained pipeline and exposes /predict and /health.
"""
import os
from pathlib import Path

import pandas as pd
import pickle

# Default path to model (override with MODEL_PATH env var)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "attrition_model.pkl"

_model = None


def get_model():
    global _model
    if _model is None:
        path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}. Run the pipeline first: python run_pipeline.py")
        with open(path, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict_one(features: dict) -> dict:
    """
    Predict attrition for a single employee.
    features: dict with keys matching training columns (no 'target').
    Returns: { "prediction": 0|1, "probability_leave": float, "probability_stay": float }
    """
    pipe = get_model()
    preprocess = pipe.named_steps.get("preprocess")
    cols = list(preprocess.feature_names_in_) if preprocess is not None and hasattr(preprocess, "feature_names_in_") else list(features.keys())
    df = pd.DataFrame([features]).reindex(columns=cols)
    proba = pipe.predict_proba(df)[0]
    pred = int(pipe.predict(df)[0])
    return {
        "prediction": pred,
        "prediction_label": "Leave" if pred == 1 else "Stay",
        "probability_stay": round(float(proba[0]), 4),
        "probability_leave": round(float(proba[1]), 4),
    }
