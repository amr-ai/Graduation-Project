"""
Pydantic v2 models for the Data Profiler.

These models define the contract between the Profiler node and all
downstream agents in the cleaning pipeline. Every LLM node receives
DatasetProfile.model_dump_json() as its structured input.
"""

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Metadata profile for a single DataFrame column."""

    name: str = Field(description="Column name")
    dtype: str = Field(description="Pandas dtype as string")
    inferred_type: str = Field(
        description="Inferred semantic type: numeric, categorical, text, date, id, boolean, or unknown"
    )
    missing_count: int = Field(description="Number of missing/null/placeholder values")
    missing_pct: float = Field(description="Percentage of missing values (0-100)")
    placeholder_count: int = Field(
        default=0,
        description="Count of known placeholder strings treated as missing (UNKNOWN, ERROR, N/A, etc.)",
    )
    unique_count: int = Field(description="Number of unique non-null values")
    sample_values: list = Field(description="Up to 5 sample values from the column")
    type_mismatch_flag: bool = Field(
        description="True if the column contains values inconsistent with its inferred dtype"
    )
    outlier_flag: bool = Field(
        description="True if the column contains statistical outliers (IQR method)"
    )
    numeric_stats: dict | None = Field(
        default=None,
        description="Numeric statistics (min, max, mean, std, skewness) or None if non-numeric",
    )


class DatasetProfile(BaseModel):
    """Full metadata profile for an entire DataFrame."""

    total_rows: int = Field(description="Total number of rows")
    total_cols: int = Field(description="Total number of columns")
    duplicate_rows_count: int = Field(description="Number of fully duplicated rows")
    columns: list[ColumnProfile] = Field(description="Per-column profiles")
    relationships: list = Field(
        default=[],
        description="Detected column relationships (e.g., arithmetic, hierarchical)",
    )
