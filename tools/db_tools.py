"""
PostgreSQL connection and data persistence utilities.

Uses environment variables for configuration:
    PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD

All functions raise on connection/query failure — caller handles errors.
"""

from __future__ import annotations

import os
import re

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.sql import quoted_name


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _safe_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: '{name}'")
    return name


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
    safe = _safe_identifier(table_name)
    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql(safe, conn, if_exists="replace", index=False)


def load_df_from_pg(table_name: str) -> pd.DataFrame:
    safe = _safe_identifier(table_name)
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {quoted_name(safe, False)}"), conn)


def get_schema_info(table_name: str) -> str:
    safe = _safe_identifier(table_name)
    engine = get_engine()
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_name = :t "
                "ORDER BY ordinal_position"
            ),
            {"t": safe},
        ).fetchall()
        row_count = conn.execute(
            text(f"SELECT COUNT(*) FROM {quoted_name(safe, False)}")
        ).scalar()
        sample = pd.read_sql(
            text(f"SELECT * FROM {quoted_name(safe, False)} LIMIT 5"), conn
        )

    parts = [
        f"Table: {table_name}  |  Rows: {row_count}  |  Columns: {len(cols)}"
    ]
    for c in cols:
        nullable = "YES" if c[2] == "YES" else "NO"
        parts.append(f"  - {c[0]}: {c[1]} (nullable={nullable})")
    parts.append(f"\nSample ({len(sample)} rows):\n{sample.to_string()}")
    return "\n".join(parts)
