"""
End-to-end HR analytics ML pipeline with SQL-first data access.
Uses real datasets when available (e.g. Saudi Employee Attrition from Mendeley).
"""
import argparse
from pathlib import Path

from src.data_loader import load_hr_csv, is_ibm_format, get_target_column
from src.db import get_connection, init_schema, load_csv_into_db, run_query_file, run_query
from src.train import train_and_evaluate
from src.visualize import generate_all_plots

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description="HR Analytics ML Pipeline (real data + SQL)")
    parser.add_argument("--data", type=str, default=None, help="Path to HR attrition CSV (optional)")
    parser.add_argument("--db", type=str, default="hr_analytics.db", help="SQLite database path")
    parser.add_argument("--model", type=str, default=None, help="Path to save trained model pickle")
    parser.add_argument("--no-train", action="store_true", help="Only load data and run SQL analytics")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) Load data: prefer real dataset in data/, else provided path, else fallback download
    print("Loading HR dataset...")
    df, source = load_hr_csv(args.data)
    print(f"  Source: {source} | Rows: {len(df)} | Columns: {list(df.columns)[:8]}...")

    if source == "fallback_download":
        print("  Note: Using benchmark dataset (fallback). For real survey data, add the Saudi")
        print("  Employee Attrition CSV to the 'data/' folder. See README.")

    # 2) SQLite: create DB and load CSV into relational table
    db_path = PROJECT_ROOT / args.db
    conn = get_connection(db_path)
    init_schema(conn)
    load_csv_into_db(conn, df)
    print(f"  Loaded into SQLite: {db_path}")

    # 3) Run analytics queries (SQL) — when schema matches IBM-style
    dept_df = None
    if is_ibm_format(df):
        print("\n--- Analytics (SQL) ---")
        try:
            sql = """
            SELECT Department, COUNT(*) AS total_employees,
                   SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attritions,
                   ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_pct
            FROM hr_raw GROUP BY Department ORDER BY attrition_pct DESC
            """
            dept_df = run_query(conn, sql)
            print(dept_df.to_string())
        except Exception as e:
            print(f"  [Warning] Skip analytics due to SQL error: {e}")

        # 4) ML feature set from SQL
        ml_df = run_query_file(conn, "queries_ml_features.sql")
        print(f"\n  ML feature set from SQL: {ml_df.shape}")
    else:
        # Generic: SELECT * and encode target in Python (e.g. Saudi or other real CSV)
        target_col = get_target_column(df)
        ml_df = run_query(conn, "SELECT * FROM hr_raw")
        # Binary target
        if ml_df[target_col].dtype == object or ml_df[target_col].astype(str).str.lower().isin(["yes", "1", "true"]).any():
            ml_df["target"] = (ml_df[target_col].astype(str).str.lower().isin(["yes", "1", "true"])).astype(int)
        else:
            ml_df["target"] = ml_df[target_col].astype(int)
        ml_df = ml_df.drop(columns=[target_col], errors="ignore")
        print(f"  ML feature set from SQL (SELECT *): {ml_df.shape}")

    conn.close()

    # 5) Train model
    if args.no_train:
        print("\nSkipping training (--no-train).")
        return

    if "target" not in ml_df.columns:
        tc = get_target_column(df)
        ml_df["target"] = (df[tc].astype(str).str.lower() == "yes").astype(int)

    print("\n--- Training model ---")
    model_path = args.model or str(OUTPUT_DIR / "attrition_model.pkl")
    metrics, pipe = train_and_evaluate(ml_df, model_path=model_path)
    print(metrics["classification_report"])
    print("CV F1 (weighted):", round(metrics["cv_f1_weighted_mean"], 4), "+/-", round(metrics["cv_f1_weighted_std"], 4))
    print(f"Model saved: {model_path}")

    # 6) Generate figures for GitHub/portfolio
    print("\n--- Generating figures ---")
    try:
        feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out()) if pipe else None
        target_counts = {"Stay": int((ml_df["target"] == 0).sum()), "Leave": int((ml_df["target"] == 1).sum())}
        generate_all_plots(
            dept_df=dept_df,
            confusion_matrix=metrics["confusion_matrix"],
            pipe=pipe,
            feature_names=feature_names,
            target_counts=target_counts,
            output_dir=OUTPUT_DIR,
        )
    except Exception as e:
        print(f"  [Warning] Could not generate some figures: {e}")


if __name__ == "__main__":
    main()
