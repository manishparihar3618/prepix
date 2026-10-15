"""
Schema validation and structural verification utilities for Prepix.

This module provides `schema_report()` to validate a pandas DataFrame against
a user-defined schema, checking required columns, data types, nullability,
and uniqueness constraints without modifying the underlying dataset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from .quality import _validate_dataframe


def _normalize_type_name(t: Any) -> str:
    """Normalize a Python type, numpy type, or string into a standard type token."""
    if isinstance(t, str):
        t_clean = t.strip().lower()
        if t_clean in ("string", "str", "text", "object"):
            return "string"
        if t_clean in ("numeric", "number"):
            return "numeric"
        if t_clean in ("integer", "int", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
            return "integer"
        if t_clean in ("float", "double", "float64", "float32", "float16"):
            return "float"
        if t_clean in ("boolean", "bool"):
            return "boolean"
        if t_clean in ("datetime", "datetime64", "timestamp", "date"):
            return "datetime"
        if t_clean in ("timedelta", "timedelta64", "duration"):
            return "timedelta"
        if t_clean in ("category", "categorical"):
            return "category"
        return t_clean

    if t in (str,):
        return "string"
    if t in (int, np.integer):
        return "integer"
    if t in (float, np.floating):
        return "float"
    if t in (bool, np.bool_):
        return "boolean"

    return str(t).lower()


def _check_type_match(series: pd.Series, expected_type_norm: str) -> bool:
    """Check if a series matches the normalized expected type."""
    if expected_type_norm in ("any", "*"):
        return True

    if expected_type_norm == "numeric":
        return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))

    if expected_type_norm == "integer":
        return bool(pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series))

    if expected_type_norm == "float":
        return bool(pd.api.types.is_float_dtype(series))

    if expected_type_norm == "string":
        return bool(
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
        )

    if expected_type_norm == "boolean":
        return bool(pd.api.types.is_bool_dtype(series))

    if expected_type_norm == "datetime":
        return bool(pd.api.types.is_datetime64_any_dtype(series))

    if expected_type_norm == "timedelta":
        return bool(pd.api.types.is_timedelta64_dtype(series))

    if expected_type_norm == "category":
        return bool(isinstance(series.dtype, pd.CategoricalDtype))

    # Exact string dtype comparison fallback
    series_dtype_str = str(series.dtype).lower()
    return expected_type_norm == series_dtype_str


def schema_report(
    df: pd.DataFrame,
    schema: Dict[str, Union[str, type, Dict[str, Any]]],
    allow_unexpected: bool = True,
) -> pd.DataFrame:
    """
    Validate a pandas DataFrame against a user-defined schema.

    Checks required columns, data types, non-null requirements, and uniqueness
    constraints, returning a detailed diagnostic report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to validate.
    schema : dict
        Mapping of column names to expected specifications. Specifications can be:
        - A type string (e.g., ``'string'``, ``'numeric'``, ``'integer'``, ``'float'``,
          ``'boolean'``, ``'datetime'``, ``'category'``)
        - A Python/numpy type (e.g., ``int``, ``float``, ``str``, ``bool``)
        - A detailed configuration dictionary with keys:
            - ``'type'``: expected type (default: ``'any'``)
            - ``'nullable'``: bool, whether null values are permitted (default: True)
            - ``'unique'``: bool, whether all values must be unique (default: False)
            - ``'required'``: bool, whether the column must be present (default: True)
    allow_unexpected : bool, default=True
        Whether columns present in the DataFrame but not specified in ``schema``
        are permitted. If False, unexpected columns are flagged with status ``'FAIL'``.

    Returns
    -------
    pandas.DataFrame
        Validation report containing:
        - Column: Name of the column
        - Expected Type: Type expected by the schema
        - Actual Type: Dtype observed in the DataFrame
        - Type Match: True if types conform, False otherwise
        - Nullable: True if nulls are allowed, False if non-null required
        - Null Count: Number of missing values observed
        - Unique Required: True if uniqueness is enforced, False otherwise
        - Duplicate Count: Number of duplicate values observed
        - Status: Overall check result (``'PASS'``, ``'FAIL'``, or ``'WARN'``)
        - Details: Human-readable diagnostic description

    Notes
    -----
    The input DataFrame is not modified. No columns or data types are altered.
    """
    _validate_dataframe(df)

    if not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary.")

    if not schema:
        raise ValueError("schema cannot be empty.")

    if not isinstance(allow_unexpected, bool):
        raise TypeError("allow_unexpected must be a boolean.")

    n_rows = len(df)
    report_rows: List[Dict[str, Any]] = []

    # 1. Process expected columns in schema
    for col, spec in schema.items():
        if isinstance(spec, (str, type)):
            expected_type_raw = str(spec.__name__) if isinstance(spec, type) else str(spec)
            expected_type_norm = _normalize_type_name(spec)
            nullable = True
            unique_required = False
            required = True
        elif isinstance(spec, dict):
            raw_type = spec.get("type", "any")
            expected_type_raw = str(raw_type.__name__) if isinstance(raw_type, type) else str(raw_type)
            expected_type_norm = _normalize_type_name(raw_type)
            nullable = bool(spec.get("nullable", True))
            unique_required = bool(spec.get("unique", False))
            required = bool(spec.get("required", True))
        else:
            raise TypeError(
                f"Specification for column '{col}' must be a type string, type, or dict, got {type(spec).__name__}."
            )

        # Check if column exists in DataFrame
        if col not in df.columns:
            if required:
                report_rows.append(
                    {
                        "Column": col,
                        "Expected Type": expected_type_raw,
                        "Actual Type": "Missing",
                        "Type Match": False,
                        "Nullable": nullable,
                        "Null Count": 0,
                        "Unique Required": unique_required,
                        "Duplicate Count": 0,
                        "Status": "FAIL",
                        "Details": "Required column missing from DataFrame",
                    }
                )
            else:
                report_rows.append(
                    {
                        "Column": col,
                        "Expected Type": expected_type_raw,
                        "Actual Type": "Missing",
                        "Type Match": True,
                        "Nullable": nullable,
                        "Null Count": 0,
                        "Unique Required": unique_required,
                        "Duplicate Count": 0,
                        "Status": "PASS",
                        "Details": "Optional column not present",
                    }
                )
            continue

        series = df[col]
        # In case of duplicate column names, select the first series
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        actual_type = str(series.dtype)
        type_match = _check_type_match(series, expected_type_norm)

        null_count = int(series.isna().sum())
        null_violation = (not nullable) and (null_count > 0)

        # Unique check
        if unique_required and n_rows > 0:
            dup_count = int(series.duplicated().sum())
            unique_violation = dup_count > 0
        else:
            dup_count = 0
            unique_violation = False

        # Determine overall column status
        failures = []
        if not type_match:
            failures.append(f"Expected '{expected_type_raw}', found '{actual_type}'")
        if null_violation:
            failures.append(f"Found {null_count} null values in non-nullable column")
        if unique_violation:
            failures.append(f"Found {dup_count} duplicate values in unique column")

        if failures:
            status = "FAIL"
            details = "; ".join(failures)
        else:
            status = "PASS"
            details = "All checks passed"

        report_rows.append(
            {
                "Column": col,
                "Expected Type": expected_type_raw,
                "Actual Type": actual_type,
                "Type Match": type_match,
                "Nullable": nullable,
                "Null Count": null_count,
                "Unique Required": unique_required,
                "Duplicate Count": dup_count,
                "Status": status,
                "Details": details,
            }
        )

    # 2. Process unexpected columns
    schema_cols_set = set(schema.keys())
    for col in df.columns:
        if col not in schema_cols_set:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            actual_type = str(series.dtype)

            if allow_unexpected:
                status = "PASS"
                details = "Unexpected column present (allowed)"
            else:
                status = "FAIL"
                details = "Unexpected column not permitted by schema"

            report_rows.append(
                {
                    "Column": col,
                    "Expected Type": "None",
                    "Actual Type": actual_type,
                    "Type Match": False,
                    "Nullable": True,
                    "Null Count": int(series.isna().sum()),
                    "Unique Required": False,
                    "Duplicate Count": int(series.duplicated().sum()) if n_rows > 0 else 0,
                    "Status": status,
                    "Details": details,
                }
            )

    output_columns = [
        "Column",
        "Expected Type",
        "Actual Type",
        "Type Match",
        "Nullable",
        "Null Count",
        "Unique Required",
        "Duplicate Count",
        "Status",
        "Details",
    ]

    return pd.DataFrame(report_rows, columns=output_columns)
