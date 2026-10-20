"""
Unified EDA and Data Quality Report for Prepix.

This module provides the high-level `report()` function and `PrepixReport`
container object that aggregates column summaries, missing value analysis,
uniqueness/cardinality, numerical distribution statistics, duplicate records,
and synthesized data-quality findings.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .quality import (
    _validate_dataframe,
    column_summary,
    duplicate_report,
    missing_report,
    numeric_summary,
    unique_report,
)


def _format_bytes(bytes_count: int) -> str:
    """Format bytes into human-readable string (B, KB, MB, GB)."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.2f} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_count / (1024 * 1024 * 1024):.2f} GB"


class PrepixReport:
    """
    Lightweight container for comprehensive dataset quality analysis.

    Attributes
    ----------
    overview : pandas.DataFrame
        High-level summary of dataset dimensions, memory, feature breakdown,
        missing values, and duplicate records.
    issues : pandas.DataFrame
        Structured list of detected data quality issues, warnings, severity,
        and recommended actions.
    columns : pandas.DataFrame
        Per-column data types, missing counts, uniqueness, and memory.
    missing : pandas.DataFrame
        Detailed missing values and imputation suggestions.
    unique : pandas.DataFrame
        Uniqueness, cardinality classification, and identifier detection.
    numeric : pandas.DataFrame
        Descriptive statistics, distribution shape, and outlier counts.
    duplicates : pandas.DataFrame
        Duplicate row counts, percentages, and deduplication guidance.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        overview: pd.DataFrame,
        issues: pd.DataFrame,
        columns: pd.DataFrame,
        missing: pd.DataFrame,
        unique: pd.DataFrame,
        numeric: pd.DataFrame,
        duplicates: pd.DataFrame,
    ) -> None:
        self._df = df
        self.overview = overview
        self.issues = issues
        self.columns = columns
        self.missing = missing
        self.unique = unique
        self.numeric = numeric
        self.duplicates = duplicates

    def to_dict(self) -> Dict[str, Any]:
        """Convert all report sections to a Python dictionary."""
        return {
            "overview": self.overview.to_dict(orient="records"),
            "issues": self.issues.to_dict(orient="records"),
            "columns": self.columns.to_dict(orient="records"),
            "missing": self.missing.to_dict(orient="records"),
            "unique": self.unique.to_dict(orient="records"),
            "numeric": self.numeric.to_dict(orient="records"),
            "duplicates": self.duplicates.to_dict(orient="records"),
        }

    def to_json(self, path_or_buf: Optional[str] = None, indent: int = 2) -> Optional[str]:
        """
        Export all report sections as formatted JSON.

        Parameters
        ----------
        path_or_buf : str, optional
            File path to save the JSON output. If None, returns the JSON string.
        indent : int, default=2
            Indentation level for formatting.

        Returns
        -------
        str or None
            JSON string if path_or_buf is None, else None.
        """
        data = self.to_dict()
        if path_or_buf is None:
            return json.dumps(data, indent=indent, default=str)
        with open(path_or_buf, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        return None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        lines: List[str] = []
        divider = "=" * 62
        sub_divider = "-" * 62

        lines.append(divider)
        lines.append("                    PREPIX QUALITY REPORT                    ")
        lines.append(divider)

        # Overview section
        lines.append("\n1. DATASET OVERVIEW")
        lines.append(sub_divider)
        for _, row in self.overview.iterrows():
            metric = row["Metric"]
            val = row["Value"]
            lines.append(f"  - {metric:<28}: {val}")

        # Issues section
        lines.append("\n2. KEY FINDINGS & ISSUES")
        lines.append(sub_divider)
        if len(self.issues) == 0 or (len(self.issues) == 1 and self.issues.iloc[0]["Severity"] == "Info"):
            lines.append("  [INFO] No significant quality issues detected.")
        else:
            for _, row in self.issues.iterrows():
                sev = row["Severity"].upper()
                col = row["Column"]
                issue = row["Issue"]
                detail = row["Detail"]
                target = f"Column '{col}'" if col != "Dataset" else "Dataset"
                lines.append(f"  [{sev:<6}] {target}: {issue} - {detail}")

        # Actionable recommendations
        lines.append("\n3. RECOMMENDED ACTIONS")
        lines.append(sub_divider)
        if len(self.issues) == 0 or (len(self.issues) == 1 and self.issues.iloc[0]["Severity"] == "Info"):
            lines.append("  * Dataset appears healthy and ready for exploratory analysis or ML.")
        else:
            recs_seen = set()
            for _, row in self.issues.iterrows():
                rec = row["Recommendation"]
                if rec and rec not in recs_seen and row["Severity"] != "Info":
                    recs_seen.add(rec)
                    lines.append(f"  * {rec}")

        # Footer
        lines.append("\n" + divider)
        lines.append("Access individual sections via:")
        lines.append("  - report.overview    - report.issues      - report.columns")
        lines.append("  - report.missing     - report.unique      - report.numeric")
        lines.append("  - report.duplicates")
        lines.append(divider)

        return "\n".join(lines)


def report(
    df: pd.DataFrame,
    low_missing_threshold: float = 10.0,
    medium_missing_threshold: float = 30.0,
    numeric_extended: bool = True,
    dropna_unique: bool = True,
) -> PrepixReport:
    """
    Generate a unified data quality and exploratory analysis report.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    low_missing_threshold : float, default=10.0
        Missing percentage threshold for 'Low' severity.
    medium_missing_threshold : float, default=30.0
        Missing percentage threshold for 'Medium' severity.
    numeric_extended : bool, default=True
        Whether to include extended numeric statistics (skewness, kurtosis,
        IQR, zeros, negatives, outliers).
    dropna_unique : bool, default=True
        Whether to exclude missing values when counting unique values in
        both the column summary and uniqueness sub-reports, guaranteeing
        consistent unique counts across all sections of the report.

    Returns
    -------
    PrepixReport
        Report object containing overview, issues, and detailed sub-reports.

    Notes
    -----
    The input DataFrame is not modified.
    """
    _validate_dataframe(df)

    n_rows = len(df)
    n_cols = df.shape[1]

    # Generate sub-reports
    col_rep = column_summary(df, dropna=dropna_unique)
    miss_rep = missing_report(
        df,
        low_threshold=low_missing_threshold,
        medium_threshold=medium_missing_threshold,
    )
    dup_rep = duplicate_report(df)
    uniq_rep = unique_report(df, dropna=dropna_unique)
    num_rep = numeric_summary(df, extended=numeric_extended)

    # Compute high-level overview metrics
    total_cells = n_rows * n_cols
    total_missing = int(miss_rep["Missing Values"].sum()) if n_cols > 0 else 0
    missing_cells_pct = (
        round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
    )
    cols_with_missing = (
        int((miss_rep["Missing Values"] > 0).sum()) if n_cols > 0 else 0
    )

    dup_rows = int(dup_rep.iloc[0]["Duplicate Rows"]) if len(dup_rep) > 0 else 0
    dup_pct = float(dup_rep.iloc[0]["Duplicate Percentage(%)"]) if len(dup_rep) > 0 else 0.0

    # Memory usage
    try:
        total_memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        total_memory_bytes = int(df.memory_usage().sum()) if n_cols > 0 else 0

    # Dtype breakdown
    numeric_count = len(num_rep)
    cat_count = 0
    dt_count = 0
    bool_count = 0
    other_count = 0

    if n_cols > 0:
        for i in range(n_cols):
            s = df.iloc[:, i]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and not pd.api.types.is_complex_dtype(s):
                continue
            elif pd.api.types.is_bool_dtype(s):
                bool_count += 1
            elif pd.api.types.is_datetime64_any_dtype(s):
                dt_count += 1
            elif (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
            ):
                cat_count += 1
            else:
                other_count += 1

    overview_data = [
        {"Metric": "Rows", "Value": str(n_rows)},
        {"Metric": "Columns", "Value": str(n_cols)},
        {"Metric": "Total Cells", "Value": str(total_cells)},
        {"Metric": "Memory Usage", "Value": _format_bytes(total_memory_bytes)},
        {"Metric": "Numeric Columns", "Value": str(numeric_count)},
        {"Metric": "Categorical Columns", "Value": str(cat_count)},
        {"Metric": "Datetime Columns", "Value": str(dt_count)},
        {"Metric": "Boolean Columns", "Value": str(bool_count)},
        {"Metric": "Other Columns", "Value": str(other_count)},
        {"Metric": "Total Missing Cells", "Value": f"{total_missing} ({missing_cells_pct}%)"},
        {"Metric": "Columns with Missing", "Value": str(cols_with_missing)},
        {"Metric": "Duplicate Rows", "Value": f"{dup_rows} ({dup_pct}%)"},
    ]
    overview_df = pd.DataFrame(overview_data)

    # Synthesize issues & warnings
    issues_list: List[Dict[str, str]] = []

    # 1. Duplicates
    if dup_rows > 0:
        issues_list.append(
            {
                "Severity": "Medium" if dup_pct < 20.0 else "High",
                "Column": "Dataset",
                "Issue": "Duplicate Records",
                "Detail": f"{dup_rows} duplicate rows ({dup_pct}%) detected",
                "Recommendation": "Review and consider deduplicating using df.drop_duplicates()",
            }
        )

    # 2. Missing values
    if n_cols > 0:
        for _, r in miss_rep.iterrows():
            m_cnt = r["Missing Values"]
            m_pct = r["Missing Percentage(%)"]
            col_name = str(r["Column"])
            status = r["Status"]
            if status == "High":
                issues_list.append(
                    {
                        "Severity": "High",
                        "Column": col_name,
                        "Issue": "High Missingness",
                        "Detail": f"{m_cnt} missing values ({m_pct}%)",
                        "Recommendation": f"Investigate cause of missingness in '{col_name}'; review before imputation or drop",
                    }
                )
            elif status == "Medium":
                issues_list.append(
                    {
                        "Severity": "Medium",
                        "Column": col_name,
                        "Issue": "Moderate Missingness",
                        "Detail": f"{m_cnt} missing values ({m_pct}%)",
                        "Recommendation": f"Consider median/mode imputation or indicator flag for '{col_name}'",
                    }
                )

    # 3. Uniqueness / Cardinality issues
    if n_cols > 0:
        for i in range(n_cols):
            col_name = df.columns[i]
            r = uniq_rep.iloc[i]
            card = r["Cardinality"]
            u_cnt = r["Unique Values"]
            s = df.iloc[:, i]

            if card == "Constant" and n_rows > 1:
                issues_list.append(
                    {
                        "Severity": "Low",
                        "Column": str(col_name),
                        "Issue": "Zero Variance (Constant)",
                        "Detail": f"Contains single constant value across all {n_rows} rows",
                        "Recommendation": f"Consider dropping constant column '{col_name}' prior to ML modeling",
                    }
                )
            elif card == "Very High (Unique)" and n_rows > 5:
                # Check if non-float
                if not pd.api.types.is_float_dtype(s):
                    issues_list.append(
                        {
                            "Severity": "Low",
                            "Column": str(col_name),
                            "Issue": "Potential Identifier",
                            "Detail": f"100% unique values ({u_cnt}/{n_rows})",
                            "Recommendation": f"Verify whether '{col_name}' is a primary key/ID; exclude from predictive features",
                        }
                    )

    # 4. Numeric distribution & outlier issues
    if len(num_rep) > 0 and numeric_extended and "Outliers" in num_rep.columns:
        for _, r in num_rep.iterrows():
            col_name = str(r["Column"])
            outliers = r.get("Outliers", 0)
            skew = r.get("Skewness", np.nan)

            if pd.notna(outliers) and outliers > 0:
                issues_list.append(
                    {
                        "Severity": "Low",
                        "Column": col_name,
                        "Issue": "Potential Outliers",
                        "Detail": f"{int(outliers)} values outside 1.5x IQR boundaries",
                        "Recommendation": f"Inspect distribution and consider robust scaling or clipping for '{col_name}'",
                    }
                )
            if pd.notna(skew) and abs(skew) > 2.0:
                direction = "right/positive" if skew > 0 else "left/negative"
                issues_list.append(
                    {
                        "Severity": "Low",
                        "Column": col_name,
                        "Issue": "Highly Skewed Distribution",
                        "Detail": f"Skewness of {skew:.2f} ({direction}-skewed)",
                        "Recommendation": f"Consider log or power transformation on '{col_name}' for linear models",
                    }
                )

    if not issues_list:
        issues_list.append(
            {
                "Severity": "Info",
                "Column": "Dataset",
                "Issue": "Clean Dataset",
                "Detail": "No quality issues detected",
                "Recommendation": "Dataset is ready for EDA or preprocessing",
            }
        )

    issues_df = pd.DataFrame(issues_list)

    return PrepixReport(
        df=df,
        overview=overview_df,
        issues=issues_df,
        columns=col_rep,
        missing=miss_rep,
        unique=uniq_rep,
        numeric=num_rep,
        duplicates=dup_rep,
    )
