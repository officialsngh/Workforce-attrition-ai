"""
Data loader for HR analytics pipeline.
Prefers real datasets; can download a fallback for quick runs.
"""
from pathlib import Path
import pandas as pd

# Real dataset: Saudi Employee Attrition (Mendeley). Place downloaded CSV here.
# https://data.mendeley.com/datasets/6z2hty8php/1
REAL_DATASET_DIR = Path(__file__).resolve().parents[1] / "data"
FALLBACK_URL = (
    "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/"
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


def _find_real_dataset() -> Path | None:
    """Look for Saudi (or other real) HR CSV in data/."""
    if not REAL_DATASET_DIR.exists():
        return None
    for f in REAL_DATASET_DIR.glob("*.csv"):
        name = f.name.lower()
        # Saudi dataset filenames often contain "Employee" and "Attrition"
        if "attrition" in name or "employee" in name or "hr_" in name or "human_resource" in name:
            return f
    return None


def load_hr_csv(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """
    Load HR attrition CSV. Prefer real data in data/; else use provided path or download fallback.
    Returns (dataframe, source_label).
    """
    path = None
    if csv_path and Path(csv_path).exists():
        path = Path(csv_path)
        label = "user_provided"
    else:
        found = _find_real_dataset()
        if found:
            path = found
            label = "real_data"
        else:
            label = "fallback_download"

    if path is not None:
        df = pd.read_csv(path)
        return df, label

    # Fallback: download IBM-style dataset (benchmark; not real survey data)
    df = pd.read_csv(FALLBACK_URL)
    return df, label


def get_target_column(df: pd.DataFrame) -> str:
    """Return the attrition/target column name (case-insensitive)."""
    for c in df.columns:
        if c.lower() == "attrition":
            return c
    raise ValueError("No 'Attrition' column found in the dataset.")


def is_ibm_format(df: pd.DataFrame) -> bool:
    """True if dataframe has IBM HR dataset column set."""
    required = {"Age", "Attrition", "Department", "JobRole", "MonthlyIncome"}
    return required.issubset(set(df.columns))
