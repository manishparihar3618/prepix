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


def column_summary(df):
   """
   Generate a column-wise summary of the dataset.
   """