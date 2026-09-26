"""
SQLite database setup and query execution for HR analytics.
All feature and analytics extraction goes through SQL.
"""
import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
SCHEMA_FILE = SQL_DIR / "schema.sql"


def get_connection(db_path: str | Path = "hr_analytics.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    if SCHEMA_FILE.exists():
        with open(SCHEMA_FILE, encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()


# SQLite limit on bound parameters per statement (conservative)
_SQLITE_MAX_VARS = 999


def load_csv_into_db(conn: sqlite3.Connection, df: pd.DataFrame, table: str = "hr_raw") -> None:
    """Create/replace table from dataframe so any real CSV schema is supported.
    Uses chunked inserts to stay under SQLite's parameter limit."""
    n_cols = len(df.columns)
    chunksize = max(1, _SQLITE_MAX_VARS // n_cols)
    df.to_sql(
        table, conn, if_exists="replace", index=False,
        method="multi", chunksize=chunksize
    )
    conn.commit()


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def run_query_file(conn: sqlite3.Connection, filename: str) -> pd.DataFrame:
    path = SQL_DIR / filename
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    return pd.read_sql_query(sql, conn)
