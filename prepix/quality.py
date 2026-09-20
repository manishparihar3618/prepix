"""
Quality and EDA utilities for Prepix.

This module contains utilities for inspecting dataset quality,
including missing-value analysis and column-level summaries.
"""
import pandas as pd


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row.")


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

    unique_values = series.nunique(dropna=True)

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

    if not 0 <= low_threshold <= 100:
        raise ValueError("low_threshold must be between 0 and 100.")

    if not 0 <= medium_threshold <= 100:
        raise ValueError("medium_threshold must be between 0 and 100.")

    if low_threshold > medium_threshold:
        raise ValueError(
            "low_threshold cannot be greater than medium_threshold."
        )

    missing_count = df.isna().sum()

    missing_percentage = (
        (missing_count / len(df)) * 100
    ).round(2)

    report = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Values": missing_count.to_numpy(),
            "Missing Percentage(%)": missing_percentage.to_numpy(),
        }
    )

    def get_status(percent: float) -> str:
        if percent == 0:
            return "No Missing"
        elif percent <= low_threshold:
            return "Low"
        elif percent <= medium_threshold:
            return "Medium"
        else:
            return "High"

    report["Status"] = report["Missing Percentage(%)"].apply(get_status)

    def get_suggestion(column: str, percent: float) -> str:
        if percent == 0:
            return "No Action Needed"

        series = df[column]

        if percent > medium_threshold:
            return "Review Before Imputation"

        if pd.api.types.is_numeric_dtype(series):
            return "Consider Median Imputation"

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            return "Consider Mode Imputation"

        return "Review Column"

    report["Suggestion"] = [
        get_suggestion(column, percentage)
        for column, percentage in zip(
            report["Column"],
            report["Missing Percentage(%)"],
        )
    ]

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

    summary = []

    for column in df.columns:
        series = df[column]

        missing = int(series.isna().sum())
        missing_percentage = round(
            (missing / len(df)) * 100,
            2,
        )

        unique = int(series.nunique(dropna=True))

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