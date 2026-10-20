"""
Quality and EDA utilities for Prepix.

This module contains utilities for inspecting dataset quality,
including missing-value analysis, column summaries, duplicate detection,
uniqueness/cardinality analysis, and numerical feature summaries.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input is a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")


def _get_suggested_type(series: pd.Series, unique_count: Optional[int] = None) -> str:
    """
    Infer a high-level suggested data type for a column.

    This is a heuristic classification, not a conversion.

    Parameters
    ----------
    series : pandas.Series
        The column series to inspect.
    unique_count : int, optional
        Precomputed unique value count to avoid redundant scans.
    """
    if pd.api.types.is_bool_dtype(series):
        return "Binary"

    if pd.api.types.is_numeric_dtype(series):
        # Numeric columns remain numeric even if they contain
        # only two unique values, such as 0 and 1.
        return "Numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    if pd.api.types.is_timedelta64_dtype(series):
        return "Timedelta"

    if isinstance(series.dtype, pd.PeriodDtype):
        return "Period"

    if unique_count is None:
        try:
            unique_values = int(series.nunique(dropna=True))
        except TypeError:
            try:
                unique_values = int(len(set(map(repr, series.dropna()))))
            except Exception:
                unique_values = -1
    else:
        unique_values = unique_count

    if unique_values == 2:
        return "Categorical (2 values)"

    return "Categorical"


def missing_report(
    df: pd.DataFrame,
    low_threshold: float = 10.0,
    medium_threshold: float = 30.0,
    only_missing: bool = False,
) -> pd.DataFrame:
    """
    Generate a missing-value report for a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    low_threshold : float, default=10.0
        Maximum missing percentage considered "Low".

    medium_threshold : float, default=30.0
        Maximum missing percentage considered "Medium".
        Values above this threshold are classified as "High".

    only_missing : bool, default=False
        If True, only columns with at least one missing value are included.

    Returns
    -------
    pandas.DataFrame
        Report containing:
        - Column
        - Missing Values
        - Missing Percentage(%)
        - Status
        - Suggestion

    Notes
    -----
    The suggestions are heuristic recommendations and should
    not be treated as automatic decisions.

    The input DataFrame is not modified.
    """
    _validate_dataframe(df)

    if isinstance(low_threshold, bool) or not isinstance(low_threshold, (int, float)):
        raise TypeError("low_threshold must be a numeric value.")
    if np.isnan(low_threshold):
        raise ValueError("low_threshold cannot be NaN.")

    if isinstance(medium_threshold, bool) or not isinstance(medium_threshold, (int, float)):
        raise TypeError("medium_threshold must be a numeric value.")
    if np.isnan(medium_threshold):
        raise ValueError("medium_threshold cannot be NaN.")

    if not 0 <= low_threshold <= 100:
        raise ValueError("low_threshold must be between 0 and 100.")

    if not 0 <= medium_threshold <= 100:
        raise ValueError("medium_threshold must be between 0 and 100.")

    if low_threshold > medium_threshold:
        raise ValueError(
            "low_threshold cannot be greater than medium_threshold."
        )

    if not isinstance(only_missing, bool):
        raise TypeError("only_missing must be a boolean.")

    output_columns = [
        "Column",
        "Missing Values",
        "Missing Percentage(%)",
        "Status",
        "Suggestion",
    ]

    if df.shape[1] == 0:
        return pd.DataFrame(columns=output_columns)

    n_rows = len(df)
    missing_count = df.isna().sum()

    if n_rows == 0:
        missing_percentage = pd.Series(0.0, index=df.columns)
    else:
        missing_percentage = ((missing_count / n_rows) * 100).round(2)

    def get_status(percent: float) -> str:
        if percent == 0:
            return "No Missing"
        elif percent <= low_threshold:
            return "Low"
        elif percent <= medium_threshold:
            return "Medium"
        else:
            return "High"

    def get_suggestion(series: pd.Series, percent: float) -> str:
        if percent == 0:
            return "No Action Needed"

        if percent > medium_threshold:
            return "Review Before Imputation"

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return "Consider Median Imputation"

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            return "Consider Mode Imputation"

        return "Review Column"

    # Use positional indexing (iloc) to safely handle duplicate column names
    suggestions = [
        get_suggestion(df.iloc[:, i], missing_percentage.iloc[i])
        for i in range(df.shape[1])
    ]

    report = pd.DataFrame(
        {
            "Column": list(df.columns),
            "Missing Values": missing_count.to_numpy(),
            "Missing Percentage(%)": missing_percentage.to_numpy(),
            "Status": [get_status(p) for p in missing_percentage],
            "Suggestion": suggestions,
        }
    )

    if only_missing:
        report = report[report["Missing Values"] > 0]

    return (
        report
        .sort_values(
            by="Missing Percentage(%)",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def column_summary(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Generate a column-wise summary of a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    dropna : bool, default=True
        Whether to exclude NaN/null values when counting unique values.
        If True (default), matches pandas Series.nunique(dropna=True).
        If False, NaN/null is counted as a distinct category.

    Returns
    -------
    pandas.DataFrame
        Summary containing:
        - Column
        - Data Type
        - Missing Values
        - Missing Percentage(%)
        - Unique Values
        - Memory(Bytes)
        - Suggested Type

    Notes
    -----
    The suggested type is a heuristic classification.
    It does not modify the original DataFrame.
    """
    _validate_dataframe(df)

    if not isinstance(dropna, bool):
        raise TypeError("dropna must be a boolean.")

    if df.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "Column",
                "Data Type",
                "Missing Values",
                "Missing Percentage(%)",
                "Unique Values",
                "Memory(Bytes)",
                "Suggested Type",
            ]
        )

    n_rows = len(df)
    summary = []

    for i in range(df.shape[1]):
        column = df.columns[i]
        series = df.iloc[:, i]

        missing = int(series.isna().sum())
        missing_percentage = (
            round((missing / n_rows) * 100, 2) if n_rows > 0 else 0.0
        )

        try:
            unique = int(series.nunique(dropna=dropna))
        except TypeError:
            try:
                target = series if not dropna else series.dropna()
                unique = int(len(set(map(repr, target))))
            except Exception:
                unique = -1

        # deep=True gives a more useful estimate for object/string data.
        memory = int(series.memory_usage(index=False, deep=True))

        summary.append(
            {
                "Column": column,
                "Data Type": str(series.dtype),
                "Missing Values": missing,
                "Missing Percentage(%)": missing_percentage,
                "Unique Values": unique,
                "Memory(Bytes)": memory,
                "Suggested Type": _get_suggested_type(series, unique_count=unique),
            }
        )

    return pd.DataFrame(summary)


def duplicate_report(
    df: pd.DataFrame,
    subset: Optional[Union[Sequence[str], str]] = None,
    keep: Union[str, bool] = "first",
    return_records: bool = False,
) -> pd.DataFrame:
    """
    Generate a summary report of duplicate rows in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    subset : sequence of column names or str, optional
        Only consider certain columns for identifying duplicates.
        If None, all columns are used.

    keep : {'first', 'last', False}, default='first'
        Determines which duplicates (if any) to mark:
        - 'first': Mark duplicates as True except for the first occurrence.
        - 'last': Mark duplicates as True except for the last occurrence.
        - False: Mark all duplicates as True.

    return_records : bool, default=False
        If True, returns a DataFrame containing the duplicate records.
        If False (default), returns the standard duplicate summary report.

    Returns
    -------
    pandas.DataFrame
        Report containing:
        - Total Rows
        - Duplicate Rows
        - Duplicate Percentage(%)
        - Unique Rows
        - Status
        - Suggestion

    Notes
    -----
    A row is considered a duplicate if an identical row occurs earlier
    in the DataFrame (following pandas `df.duplicated()` semantics).

    The input DataFrame is not modified.
    """
    _validate_dataframe(df)

    if not isinstance(return_records, bool):
        raise TypeError("return_records must be a boolean.")

    if keep not in ("first", "last", False):
        raise ValueError("keep must be one of 'first', 'last', or False.")

    if subset is not None:
        if isinstance(subset, str):
            subset_cols = [subset]
        elif hasattr(subset, "__iter__"):
            subset_cols = list(subset)
        else:
            raise TypeError("subset must be a string, list of strings, or None.")

        missing_cols = [col for col in subset_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
    else:
        subset_cols = None

    total_rows = len(df)
    if total_rows == 0:
        duplicate_rows = 0
        duplicate_percentage = 0.0
        unique_rows = 0
        status = "No Duplicates"
        suggestion = "No action required"
        dup_mask = pd.Series([], dtype=bool)
    else:
        try:
            dup_mask = df.duplicated(subset=subset_cols, keep=keep)
            duplicate_rows = int(dup_mask.sum())
        except TypeError:
            try:
                # Map unhashable objects (lists, dicts, sets) to their string representations
                map_fn = getattr(df, "map", getattr(df, "applymap", None))
                if map_fn is not None:
                    safe_df = map_fn(lambda x: repr(x) if isinstance(x, (list, dict, set)) else x)
                    dup_mask = safe_df.duplicated(subset=subset_cols, keep=keep)
                    duplicate_rows = int(dup_mask.sum())
                else:
                    dup_mask = pd.Series([False] * total_rows, index=df.index)
                    duplicate_rows = 0
            except Exception:
                dup_mask = pd.Series([False] * total_rows, index=df.index)
                duplicate_rows = 0

        duplicate_percentage = round((duplicate_rows / total_rows) * 100, 2)
        unique_rows = total_rows - duplicate_rows

        if duplicate_rows == 0:
            status = "No Duplicates"
            suggestion = "No action required"
        else:
            status = "Duplicates Found"
            suggestion = "Consider reviewing and removing duplicate records"

    if return_records:
        if total_rows == 0:
            return df.iloc[0:0].copy()
        return df.iloc[dup_mask.to_numpy()].copy()

    return pd.DataFrame(
        [
            {
                "Total Rows": total_rows,
                "Duplicate Rows": duplicate_rows,
                "Duplicate Percentage(%)": duplicate_percentage,
                "Unique Rows": unique_rows,
                "Status": status,
                "Suggestion": suggestion,
            }
        ]
    )


def unique_report(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    """
    Generate a uniqueness and cardinality report for every column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    dropna : bool, default=False
        Whether to exclude NaN/null values when counting unique values.
        If False (default), NaN/null is counted as a distinct category if present.

    Returns
    -------
    pandas.DataFrame
        Report containing:
        - Column
        - Data Type
        - Total Values
        - Unique Values
        - Unique Percentage(%)
        - Duplicate Values
        - Cardinality
        - Status
        - Suggestion

    Notes
    -----
    Cardinality classifications:
    - Constant: Exactly 1 unique value (Zero variance).
    - Very High (Unique): 100% unique values (Potential unique identifier).
    - Very High: >= 90% unique values (Near-unique / high diversity).
    - High: 50% - 90% unique values.
    - Medium: 10% - 50% unique values.
    - Low: < 10% unique values.

    The suggestions are heuristic recommendations and do not modify the input DataFrame.
    """
    _validate_dataframe(df)

    if not isinstance(dropna, bool):
        raise TypeError("dropna must be a boolean.")

    if df.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "Column",
                "Data Type",
                "Total Values",
                "Unique Values",
                "Unique Percentage(%)",
                "Duplicate Values",
                "Cardinality",
                "Status",
                "Suggestion",
            ]
        )

    n_rows = len(df)
    report_rows = []

    for i in range(df.shape[1]):
        column = df.columns[i]
        series = df.iloc[:, i]
        dtype_str = str(series.dtype)

        if n_rows == 0:
            report_rows.append(
                {
                    "Column": column,
                    "Data Type": dtype_str,
                    "Total Values": 0,
                    "Unique Values": 0,
                    "Unique Percentage(%)": 0.0,
                    "Duplicate Values": 0,
                    "Cardinality": "Empty",
                    "Status": "Empty Column",
                    "Suggestion": "No action needed",
                }
            )
            continue

        try:
            unique_count = int(series.nunique(dropna=dropna))
        except TypeError:
            try:
                target = series if not dropna else series.dropna()
                unique_count = int(len(set(map(repr, target))))
            except Exception:
                unique_count = n_rows
        unique_pct = round((unique_count / n_rows) * 100, 2)
        valid_count = int(series.count())
        duplicate_count = (n_rows - unique_count) if not dropna else max(0, valid_count - unique_count)

        col_str = str(column).lower().strip()
        is_id_name = any(
            x in col_str.replace("-", "_").split("_")
            for x in ["id", "key", "code", "num", "number"]
        ) or col_str in ["id", "key", "code", "uuid", "guid"]

        if unique_count == 0:
            cardinality = "Empty"
            status = "All Missing"
            suggestion = "Column contains only missing values; consider dropping or imputing"
        elif unique_count == 1:
            cardinality = "Constant"
            status = "Zero Variance"
            suggestion = "Consider reviewing column; constant value across all rows"
        elif unique_count == n_rows:
            cardinality = "Very High (Unique)"
            status = "Potential Identifier"
            if pd.api.types.is_float_dtype(series):
                suggestion = "Continuous numeric distribution with high uniqueness; review if feature or identifier"
            else:
                suggestion = "Review whether this column is a unique identifier"
        elif unique_pct >= 90.0:
            cardinality = "Very High"
            status = "Near-Unique"
            if is_id_name or (pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series)):
                suggestion = "Review whether this column is an identifier, sequence, or high-cardinality feature"
            elif pd.api.types.is_float_dtype(series):
                suggestion = "Continuous numeric distribution with high uniqueness"
            else:
                suggestion = "Review whether this column is an identifier or high-cardinality feature"
        elif unique_pct >= 50.0:
            cardinality = "High"
            status = "High Diversity"
            if is_id_name:
                suggestion = "High uniqueness discrete values; review whether this column is an identifier or key"
            elif pd.api.types.is_float_dtype(series):
                suggestion = "High uniqueness numeric distribution; review distribution spread"
            elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                suggestion = "High uniqueness discrete numeric values; review distribution"
            else:
                suggestion = "High cardinality categorical; review before one-hot encoding or grouping"
        elif unique_pct >= 10.0:
            cardinality = "Medium"
            status = "Moderate Diversity"
            suggestion = "Suitable for standard encoding or analysis"
        else:
            cardinality = "Low"
            if unique_count == 2:
                status = "Binary"
                suggestion = "Suitable for binary encoding"
            else:
                status = "Low Diversity"
                suggestion = "Suitable for categorical encoding"

        report_rows.append(
            {
                "Column": column,
                "Data Type": dtype_str,
                "Total Values": n_rows,
                "Unique Values": unique_count,
                "Unique Percentage(%)": unique_pct,
                "Duplicate Values": duplicate_count,
                "Cardinality": cardinality,
                "Status": status,
                "Suggestion": suggestion,
            }
        )

    return pd.DataFrame(report_rows)


def numeric_summary(
    df: pd.DataFrame,
    extended: bool = False,
    percentiles: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Generate a statistical summary for all numerical columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    extended : bool, default=False
        Whether to include extended statistical diagnostics:
        - IQR (Interquartile Range)
        - Skewness (Fisher-Pearson distribution asymmetry)
        - Kurtosis (Fisher excess peakedness)
        - Zeros (Count of zero values)
        - Negatives (Count of negative values)
        - Outliers (Count of observations beyond 1.5 * IQR)

    percentiles : sequence of float, optional
        Additional percentiles to calculate, e.g. [0.05, 0.95].
        Values must be between 0.0 and 1.0.

    Returns
    -------
    pandas.DataFrame
        Summary report containing one row per numerical column:
        - Column: Name of the numerical column
        - Data Type: String representation of the column's data type
        - Count: Number of non-null (valid) numerical observations
        - Missing Values: Number of missing/null (NaN) entries
        - Mean: Arithmetic average of valid observations
        - Median: 50th percentile of valid observations
        - Std: Sample standard deviation (ddof=1)
        - Min: Minimum valid numerical value
        - Q1: 25th percentile of valid observations
        - Q3: 75th percentile of valid observations
        - Max: Maximum valid numerical value
        - Range: Spread between Maximum and Minimum (Max - Min)
        [If extended=True: IQR, Skewness, Kurtosis, Zeros, Negatives, Outliers]
        [If percentiles: Corresponding percentile columns, e.g. P5, P95]

    Notes
    -----
    - Numerical Column Detection: Includes standard integer, float, nullable integer,
      and nullable float dtypes. Boolean and complex columns are treated as non-numerical
      and excluded from the summary.
    - Missing Values: Statistics are calculated ignoring NaN values. Columns with 0 valid
      observations return NaN for statistical measures without crashing.
    - Standard Deviation: Uses sample standard deviation with N - 1 degrees of freedom (ddof=1).
      Columns with single valid observations return NaN for Std.
    - Infinity: Infinite values (np.inf, -np.inf) are retained according to standard
      Pandas semantics and not converted to NaN.
    - Outlier Detection: Identifies observations falling outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    - The input DataFrame is not modified.
    """
    _validate_dataframe(df)

    if not isinstance(extended, bool):
        raise TypeError("extended must be a boolean.")

    validated_percentiles: List[float] = []
    if percentiles is not None:
        if not hasattr(percentiles, "__iter__") or isinstance(percentiles, (str, bytes)):
            raise TypeError("percentiles must be a sequence of floats between 0 and 1.")
        for p in percentiles:
            if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0.0 <= p <= 1.0):
                raise ValueError(f"Percentile {p} must be a number between 0 and 1.")
            validated_percentiles.append(float(p))

    base_columns = [
        "Column",
        "Data Type",
        "Count",
        "Missing Values",
        "Mean",
        "Median",
        "Std",
        "Min",
        "Q1",
        "Q3",
        "Max",
        "Range",
    ]

    output_columns = list(base_columns)
    if extended:
        output_columns.extend(["IQR", "Skewness", "Kurtosis", "Zeros", "Negatives", "Outliers"])
    for p in validated_percentiles:
        p_pct = p * 100
        p_col = f"P{int(p_pct)}" if p_pct.is_integer() else f"P{p_pct:.1f}"
        if p_col not in output_columns:
            output_columns.append(p_col)

    if df.shape[1] == 0:
        return pd.DataFrame(columns=output_columns)

    report_rows = []

    for i in range(df.shape[1]):
        column = df.columns[i]
        series = df.iloc[:, i]

        # Numerical detection: integer/float (including nullable), excluding bool and complex
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series) or pd.api.types.is_complex_dtype(series):
            continue

        valid_count = int(series.count())
        missing_count = int(series.isna().sum())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if valid_count == 0:
                mean_val = np.nan
                median_val = np.nan
                std_val = np.nan
                min_val = np.nan
                q1_val = np.nan
                q3_val = np.nan
                max_val = np.nan
                range_val = np.nan
                iqr_val = np.nan
                skew_val = np.nan
                kurt_val = np.nan
                zeros_val = 0
                negatives_val = 0
                outliers_val = 0
            else:
                mean_val = series.mean()
                median_val = series.median()
                std_val = series.std() if valid_count > 1 else np.nan
                min_val = series.min()
                q1_val = series.quantile(0.25)
                q3_val = series.quantile(0.75)
                max_val = series.max()
                try:
                    range_val = max_val - min_val
                except Exception:
                    range_val = np.nan

                if extended:
                    try:
                        iqr_val = q3_val - q1_val if (pd.notna(q3_val) and pd.notna(q1_val)) else np.nan
                    except Exception:
                        iqr_val = np.nan

                    skew_val = float(series.skew()) if valid_count >= 3 else np.nan
                    kurt_val = float(series.kurt()) if valid_count >= 4 else np.nan

                    try:
                        zeros_val = int((series == 0).sum())
                    except Exception:
                        zeros_val = 0

                    try:
                        negatives_val = int((series < 0).sum())
                    except Exception:
                        negatives_val = 0

                    if pd.notna(iqr_val) and not np.isinf(iqr_val) and iqr_val >= 0:
                        lower_bound = q1_val - 1.5 * iqr_val
                        upper_bound = q3_val + 1.5 * iqr_val
                        try:
                            outliers_val = int(((series < lower_bound) | (series > upper_bound)).sum())
                        except Exception:
                            outliers_val = 0
                    else:
                        outliers_val = 0

            row_dict: dict[str, Any] = {
                "Column": column,
                "Data Type": str(series.dtype),
                "Count": valid_count,
                "Missing Values": missing_count,
                "Mean": mean_val,
                "Median": median_val,
                "Std": std_val,
                "Min": min_val,
                "Q1": q1_val,
                "Q3": q3_val,
                "Max": max_val,
                "Range": range_val,
            }

            if extended:
                row_dict["IQR"] = iqr_val
                row_dict["Skewness"] = skew_val
                row_dict["Kurtosis"] = kurt_val
                row_dict["Zeros"] = zeros_val
                row_dict["Negatives"] = negatives_val
                row_dict["Outliers"] = outliers_val

            for p in validated_percentiles:
                p_pct = p * 100
                p_col = f"P{int(p_pct)}" if p_pct.is_integer() else f"P{p_pct:.1f}"
                row_dict[p_col] = series.quantile(p) if valid_count > 0 else np.nan

            report_rows.append(row_dict)

    if not report_rows:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(report_rows)


# Convenience imports for cross-module accessibility
from .report import PrepixReport, report  # noqa: E402, F401