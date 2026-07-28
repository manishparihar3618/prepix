"""
Quality check utilities for Prepix.
"""
import pandas as pd
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a missing value report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Missing value report with counts, percentages,
        status, and suggestions.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    ).round(2)

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_count.values,
        "Missing Percentage(%)": missing_percentage.values
    })

    def get_status(percent):
        if percent == 0:
            return "No Missing"
        elif percent <= 10:
            return "Low"
        elif percent <= 30:
            return "Medium"
        else:
            return "High"

    report["Status"] = report["Missing Percentage(%)"].apply(get_status)

    def get_suggestion(column, percent):
        if percent == 0:
            return "No Action Needed"

        if pd.api.types.is_numeric_dtype(df[column]):
            if percent <= 30:
                return "Median Imputation"
            return "Consider Dropping"

        else:
            if percent <= 30:
                return "Mode Imputation"
            return "Consider Dropping"

    report["Suggestion"] = [
        get_suggestion(col, pct)
        for col, pct in zip(
            report["Column"],
            report["Missing Percentage(%)"]
        )
    ]

    report = report.sort_values(
        by="Missing Percentage(%)",
        ascending=False
    ).reset_index(drop=True)

    return report


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
        Summary of every column.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    summary = []

    for column in df.columns:

        dtype = str(df[column].dtype)

        missing = df[column].isnull().sum()

        missing_percentage = round(
            (missing / len(df)) * 100, 2
        )

        unique = df[column].nunique(dropna=True)

        memory = df[column].memory_usage(deep=True)

        if pd.api.types.is_numeric_dtype(df[column]):
            suggested = "Numeric"

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            suggested = "Datetime"

        elif unique == 2:
            suggested = "Binary"

        else:
            suggested = "Categorical"

        summary.append({
            "Column": column,
            "Data Type": dtype,
            "Missing Values": missing,
            "Missing Percentage(%)": missing_percentage,
            "Unique Values": unique,
            "Memory(Bytes)": memory,
            "Suggested Type": suggested
        })

    return pd.DataFrame(summary)
    
