"""
PostgreSQL connection and data persistence utilities.

Uses environment variables for configuration:
    PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD

All functions raise on connection/query failure — caller handles errors.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text


def _conn_str() -> str:
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DBNAME", "smart_analyst")
    user = os.environ.get("PG_USER", "postgres")
    password = os.environ.get("PG_PASSWORD", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_engine():
    return create_engine(_conn_str())


def is_pg_configured() -> bool:
    return bool(os.environ.get("PG_HOST") and os.environ.get("PG_USER"))


def store_df_to_pg(df: pd.DataFrame, table_name: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def load_df_from_pg(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)


def get_schema_info(table_name: str) -> str:
    engine = get_engine()
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                f"SELECT column_name, data_type, is_nullable "
                f"FROM information_schema.columns "
                f"WHERE table_name = '{table_name}' "
                f"ORDER BY ordinal_position"
            )
        ).fetchall()
        row_count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar()
        sample = pd.read_sql(
            text(f"SELECT * FROM {table_name} LIMIT 5"), conn
        )

    parts = [
        f"Table: {table_name}  |  Rows: {row_count}  |  Columns: {len(cols)}"
    ]
    for c in cols:
        nullable = "YES" if c[2] == "YES" else "NO"
        parts.append(f"  - {c[0]}: {c[1]} (nullable={nullable})")
    parts.append(f"\nSample ({len(sample)} rows):\n{sample.to_string()}")
    return "\n".join(parts)
