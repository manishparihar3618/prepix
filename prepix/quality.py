"""
Quality and EDA utilities for Prepix.

This module contains utilities for inspecting dataset quality,
including missing-value analysis, column summaries, duplicate detection,
and uniqueness/cardinality analysis.
"""
import pandas as pd


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input is a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")


def _get_suggested_type(series: pd.Series) -> str:
    """
    Infer a high-level suggested data type for a column.

    This is a heuristic classification, not a conversion.
    """
    if pd.api.types.is_bool_dtype(series):
        return "Binary"

    if pd.api.types.is_numeric_dtype(series):
        # Numeric columns remain numeric even if they contain
        # only two unique values, such as 0 and 1.
        return "Numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    try:
        unique_values = series.nunique(dropna=True)
    except TypeError:
        try:
            unique_values = len(set(map(repr, series.dropna())))
        except Exception:
            unique_values = -1

    if unique_values == 2:
        return "Categorical (2 values)"

    return "Categorical"


def missing_report(
    df: pd.DataFrame,
    low_threshold: float = 10.0,
    medium_threshold: float = 30.0,
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

    if isinstance(medium_threshold, bool) or not isinstance(medium_threshold, (int, float)):
        raise TypeError("medium_threshold must be a numeric value.")

    if not 0 <= low_threshold <= 100:
        raise ValueError("low_threshold must be between 0 and 100.")

    if not 0 <= medium_threshold <= 100:
        raise ValueError("medium_threshold must be between 0 and 100.")

    if low_threshold > medium_threshold:
        raise ValueError(
            "low_threshold cannot be greater than medium_threshold."
        )

    if df.shape[1] == 0:
        return pd.DataFrame(
            columns=[
                "Column",
                "Missing Values",
                "Missing Percentage(%)",
                "Status",
                "Suggestion",
            ]
        )

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

        if pd.api.types.is_numeric_dtype(series):
            return "Consider Median Imputation"

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
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

    return (
        report
        .sort_values(
            by="Missing Percentage(%)",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a column-wise summary of a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

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
            unique = int(series.nunique(dropna=True))
        except TypeError:
            try:
                unique = int(len(set(map(repr, series.dropna()))))
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
                "Suggested Type": _get_suggested_type(series),
            }
        )

    return pd.DataFrame(summary)


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary report of duplicate rows in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

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

    total_rows = len(df)
    if total_rows == 0:
        duplicate_rows = 0
        duplicate_percentage = 0.0
        unique_rows = 0
        status = "No Duplicates"
        suggestion = "No action required"
    else:
        try:
            duplicate_rows = int(df.duplicated().sum())
        except TypeError:
            try:
                # Map unhashable objects (lists, dicts, sets) to their string representations
                map_fn = getattr(df, "map", getattr(df, "applymap", None))
                if map_fn is not None:
                    safe_df = map_fn(lambda x: repr(x) if isinstance(x, (list, dict, set)) else x)
                    duplicate_rows = int(safe_df.duplicated().sum())
                else:
                    duplicate_rows = 0
            except Exception:
                duplicate_rows = 0

        duplicate_percentage = round((duplicate_rows / total_rows) * 100, 2)
        unique_rows = total_rows - duplicate_rows

        if duplicate_rows == 0:
            status = "No Duplicates"
            suggestion = "No action required"
        else:
            status = "Duplicates Found"
            suggestion = "Consider reviewing and removing duplicate records"

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
        duplicate_count = n_rows - unique_count

        if unique_count == 1:
            cardinality = "Constant"
            status = "Zero Variance"
            suggestion = "Consider reviewing column; constant value across all rows"
        elif unique_count == n_rows:
            cardinality = "Very High (Unique)"
            status = "Potential Identifier"
            suggestion = "Review whether this column is a unique identifier"
        elif unique_pct >= 90.0:
            cardinality = "Very High"
            status = "Near-Unique"
            suggestion = "Review whether this column is an identifier or high-cardinality feature"
        elif unique_pct >= 50.0:
            cardinality = "High"
            status = "High Diversity"
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                suggestion = "Normal continuous numeric distribution"
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