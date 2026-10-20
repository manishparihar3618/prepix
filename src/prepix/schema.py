"""
Schema validation and structural verification utilities for Prepix.

This module provides `schema_report()` to validate a pandas DataFrame against
a user-defined schema, checking required columns, data types, nullability,
and uniqueness constraints without modifying the underlying dataset.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from .quality import _validate_dataframe

# Recognized standard type aliases
_TYPE_ALIASES = {
    "string": "string",
    "str": "string",
    "text": "string",
    "object": "string",
    "numeric": "numeric",
    "number": "numeric",
    "integer": "integer",
    "int": "integer",
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "uint64": "uint64",
    "uint32": "uint32",
    "uint16": "uint16",
    "uint8": "uint8",
    "float": "float",
    "double": "float",
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
    "boolean": "boolean",
    "bool": "boolean",
    "datetime": "datetime",
    "datetime64": "datetime",
    "timestamp": "datetime",
    "date": "datetime",
    "timedelta": "timedelta",
    "timedelta64": "timedelta",
    "duration": "timedelta",
    "category": "category",
    "categorical": "category",
    "any": "any",
    "*": "any",
}

_PYTHON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "float",
    bool: "boolean",
    object: "string",
    np.integer: "integer",
    np.floating: "float",
    np.bool_: "boolean",
    datetime.datetime: "datetime",
    datetime.date: "datetime",
    pd.Timestamp: "datetime",
    datetime.timedelta: "timedelta",
    pd.Timedelta: "timedelta",
}


def _normalize_type(t: Any, col: str) -> tuple[str, str]:
    """
    Normalize a type specification into (raw_display_name, normalized_token).

    Raises ValueError if the type cannot be recognized.
    Raises TypeError if the type is not a string, type, or valid specifier.
    """
    if t is None:
        return ("any", "any")

    if isinstance(t, type):
        raw_name = t.__name__
        if t in _PYTHON_TYPE_MAP:
            return (raw_name, _PYTHON_TYPE_MAP[t])
        if issubclass(t, np.integer):
            return (raw_name, "integer")
        if issubclass(t, np.floating):
            return (raw_name, "float")
        clean_name = raw_name.lower()
        if clean_name in _TYPE_ALIASES:
            return (raw_name, _TYPE_ALIASES[clean_name])
        raise ValueError(f"Unsupported data type '{raw_name}' specified for column '{col}'.")

    if isinstance(t, str):
        raw_name = t.strip()
        clean = raw_name.lower()
        if clean in _TYPE_ALIASES:
            return (raw_name, _TYPE_ALIASES[clean])
        raise ValueError(f"Unsupported data type '{raw_name}' specified for column '{col}'.")

    raise TypeError(
        f"Type specification for column '{col}' must be a string or type, got {type(t).__name__}."
    )


def _check_type_match(series: pd.Series, expected_norm: str) -> bool:
    """Check if a series matches the normalized expected type."""
    if expected_norm == "any":
        return True

    if expected_norm == "numeric":
        return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))

    if expected_norm == "integer":
        return bool(pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series))

    if expected_norm == "float":
        return bool(pd.api.types.is_float_dtype(series))

    if expected_norm == "string":
        return bool(
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
        )

    if expected_norm == "boolean":
        return bool(pd.api.types.is_bool_dtype(series))

    if expected_norm == "datetime":
        return bool(pd.api.types.is_datetime64_any_dtype(series))

    if expected_norm == "timedelta":
        return bool(pd.api.types.is_timedelta64_dtype(series))

    if expected_norm == "category":
        return bool(isinstance(series.dtype, pd.CategoricalDtype))

    # Exact string dtype match fallback (e.g., 'int32', 'float32', 'uint8')
    return str(series.dtype).lower() == expected_norm


def schema_report(
    df: pd.DataFrame,
    schema: Dict[str, Union[str, type, Dict[str, Any]]],
    allow_unexpected: bool = True,
) -> pd.DataFrame:
    """
    Validate a pandas DataFrame against a user-defined schema.

    Verifies required columns, expected data types, non-null requirements, and
    uniqueness constraints, returning a structured diagnostic DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to validate. Never modified in place.
    schema : dict
        Mapping of column names to expected specifications.
        Specifications can be:
        - A type string (e.g., ``'string'``, ``'numeric'``, ``'integer'``,
          ``'float'``, ``'boolean'``, ``'datetime'``, ``'category'``, etc.)
        - A Python/numpy type (e.g., ``str``, ``int``, ``float``, ``bool``)
        - A configuration dictionary supporting:
            - ``'type'``: str or type, expected data type
            - ``'nullable'``: bool, whether nulls are allowed (default: True)
            - ``'non_null'``: bool, alias for ``nullable=False`` when True
            - ``'unique'``: bool, whether all values must be unique (default: False)
            - ``'required'``: bool, whether the column must be present (default: True)
    allow_unexpected : bool, default=True
        Whether columns present in ``df`` but not defined in ``schema`` are
        permitted. If False, unexpected columns are flagged with status ``'FAIL'``.

    Returns
    -------
    pandas.DataFrame
        A structured validation report with the following columns:
        - ``Column``: Column name being checked
        - ``Check``: Validation check performed (``'column_presence'``,
          ``'data_type'``, ``'non_null'``, ``'unique'``, ``'unexpected_column'``)
        - ``Expected``: Expected condition or value
        - ``Actual``: Observed condition or value
        - ``Status``: Result of the check (``'PASS'`` or ``'FAIL'``)
        - ``Message``: Human-readable diagnostic explanation

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame, ``schema`` is not a dict,
        ``allow_unexpected`` is not a boolean, or schema specifications have
        invalid types.
    ValueError
        If ``schema`` is empty, contains unrecognized configuration keys,
        or specifies an unknown data type.

    Examples
    --------
    >>> import pandas as pd
    >>> import prepix as pp
    >>> df = pd.DataFrame({
    ...     "Name": ["Aman", "Priya", "Rahul"],
    ...     "Age": [25, 30, 28],
    ...     "Salary": [50000, 60000, 55000]
    ... })
    >>> schema = {
    ...     "Name": "string",
    ...     "Age": "numeric",
    ...     "Salary": {"type": "numeric", "nullable": False}
    ... }
    >>> result = pp.schema_report(df, schema=schema)
    >>> print(result)
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

    # Validate and process schema definitions
    valid_dict_keys = {"type", "nullable", "non_null", "unique", "required"}

    for col, spec in schema.items():
        if not isinstance(col, (str, int, float)):
            raise TypeError(f"Column name in schema must be a string or identifier, got {type(col).__name__}.")

        col_str = str(col)

        if isinstance(spec, (str, type)):
            has_type_check = True
            raw_type_name, norm_type = _normalize_type(spec, col_str)
            required = True
            check_nullable = False
            nullable = True
            check_unique = False
            unique_required = False
        elif isinstance(spec, dict):
            invalid_keys = set(spec.keys()) - valid_dict_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid key(s) in schema configuration for column '{col_str}': {sorted(invalid_keys)}. "
                    f"Supported keys are: {sorted(valid_dict_keys)}."
                )

            if "type" in spec:
                has_type_check = True
                raw_type_name, norm_type = _normalize_type(spec["type"], col_str)
            else:
                has_type_check = False
                raw_type_name, norm_type = ("any", "any")

            if "required" in spec:
                if not isinstance(spec["required"], bool):
                    raise TypeError(f"'required' for column '{col_str}' must be a boolean.")
                required = spec["required"]
            else:
                required = True

            if "non_null" in spec:
                if not isinstance(spec["non_null"], bool):
                    raise TypeError(f"'non_null' for column '{col_str}' must be a boolean.")
                check_nullable = True
                nullable = not spec["non_null"]
            elif "nullable" in spec:
                if not isinstance(spec["nullable"], bool):
                    raise TypeError(f"'nullable' for column '{col_str}' must be a boolean.")
                check_nullable = True
                nullable = spec["nullable"]
            else:
                check_nullable = False
                nullable = True

            if "unique" in spec:
                if not isinstance(spec["unique"], bool):
                    raise TypeError(f"'unique' for column '{col_str}' must be a boolean.")
                check_unique = True
                unique_required = spec["unique"]
            else:
                check_unique = False
                unique_required = False
        else:
            raise TypeError(
                f"Specification for column '{col_str}' must be a type string, type, or dict, got {type(spec).__name__}."
            )

        # 1. Column presence check
        col_indices = [idx for idx, c in enumerate(df.columns) if c == col]
        is_present = len(col_indices) > 0

        if not is_present:
            if required:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "column_presence",
                        "Expected": "present",
                        "Actual": "missing",
                        "Status": "FAIL",
                        "Message": f"Required column '{col_str}' is missing from DataFrame.",
                    }
                )
            else:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "column_presence",
                        "Expected": "optional",
                        "Actual": "missing",
                        "Status": "PASS",
                        "Message": f"Optional column '{col_str}' is absent.",
                    }
                )
            # Cannot perform data type, nullability, or uniqueness checks on absent column
            continue

        report_rows.append(
            {
                "Column": col_str,
                "Check": "column_presence",
                "Expected": "present",
                "Actual": "present",
                "Status": "PASS",
                "Message": (
                    f"Required column '{col_str}' is present."
                    if required
                    else f"Optional column '{col_str}' is present."
                ),
            }
        )

        # Extract column series (safe with duplicate column names)
        all_matching = [df.iloc[:, idx] for idx in col_indices]
        actual_types = [str(s.dtype) for s in all_matching]
        type_matches = [_check_type_match(s, norm_type) for s in all_matching]
        all_type_match = all(type_matches)
        actual_type_str = (
            actual_types[0]
            if len(set(actual_types)) == 1
            else ", ".join(actual_types)
        )

        null_counts = [int(s.isna().sum()) for s in all_matching]
        max_null_count = max(null_counts)

        dup_counts = [
            int(s.duplicated().sum()) if n_rows > 0 else 0
            for s in all_matching
        ]
        max_dup_count = max(dup_counts)

        # 2. Data type check
        if has_type_check:
            if all_type_match:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "data_type",
                        "Expected": raw_type_name,
                        "Actual": actual_type_str,
                        "Status": "PASS",
                        "Message": f"Data type matches expected '{raw_type_name}'.",
                    }
                )
            else:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "data_type",
                        "Expected": raw_type_name,
                        "Actual": actual_type_str,
                        "Status": "FAIL",
                        "Message": f"Expected data type '{raw_type_name}', but found '{actual_type_str}'.",
                    }
                )

        # 3. Non-null check (if requested)
        if check_nullable:
            if not nullable:
                if max_null_count == 0:
                    report_rows.append(
                        {
                            "Column": col_str,
                            "Check": "non_null",
                            "Expected": "no_missing",
                            "Actual": "0 missing",
                            "Status": "PASS",
                            "Message": f"Column '{col_str}' contains no missing values.",
                        }
                    )
                else:
                    report_rows.append(
                        {
                            "Column": col_str,
                            "Check": "non_null",
                            "Expected": "no_missing",
                            "Actual": f"{max_null_count} missing",
                            "Status": "FAIL",
                            "Message": f"Column '{col_str}' contains {max_null_count} missing value(s) in non-nullable column.",
                        }
                    )
            else:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "non_null",
                        "Expected": "nullable",
                        "Actual": f"{max_null_count} missing",
                        "Status": "PASS",
                        "Message": f"Column '{col_str}' contains {max_null_count} missing value(s) (nulls allowed).",
                    }
                )

        # 4. Uniqueness check (if requested)
        if check_unique:
            if unique_required:
                if max_dup_count == 0:
                    report_rows.append(
                        {
                            "Column": col_str,
                            "Check": "unique",
                            "Expected": "unique",
                            "Actual": "0 duplicates",
                            "Status": "PASS",
                            "Message": f"All values in column '{col_str}' are unique.",
                        }
                    )
                else:
                    report_rows.append(
                        {
                            "Column": col_str,
                            "Check": "unique",
                            "Expected": "unique",
                            "Actual": f"{max_dup_count} duplicate(s)",
                            "Status": "FAIL",
                            "Message": f"Column '{col_str}' contains {max_dup_count} duplicate value(s) in unique column.",
                        }
                    )
            else:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "unique",
                        "Expected": "any",
                        "Actual": f"{max_dup_count} duplicate(s)",
                        "Status": "PASS",
                        "Message": f"Column '{col_str}' contains {max_dup_count} duplicate value(s) (uniqueness not enforced).",
                    }
                )

    # 5. Unexpected columns check
    schema_cols_set = set(schema.keys())
    seen_unexpected = set()
    for col in df.columns:
        if col not in schema_cols_set and col not in seen_unexpected:
            seen_unexpected.add(col)
            col_str = str(col)
            if allow_unexpected:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "unexpected_column",
                        "Expected": "allowed",
                        "Actual": "present",
                        "Status": "PASS",
                        "Message": f"Unexpected column '{col_str}' is present (allowed).",
                    }
                )
            else:
                report_rows.append(
                    {
                        "Column": col_str,
                        "Check": "unexpected_column",
                        "Expected": "absent",
                        "Actual": "present",
                        "Status": "FAIL",
                        "Message": f"Unexpected column '{col_str}' is not allowed by schema.",
                    }
                )

    output_columns = [
        "Column",
        "Check",
        "Expected",
        "Actual",
        "Status",
        "Message",
    ]

    return pd.DataFrame(report_rows, columns=output_columns)
