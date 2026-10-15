import os
import pandas as pd

from prepix import (
    column_summary,
    duplicate_report,
    missing_report,
    numeric_summary,
    report,
    unique_report,
)


def run_manual_test():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data.csv")
    print(f"Loading dataset from: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("DATASET PREVIEW (First 5 Rows):")
    print("=" * 60)
    print(df.head())
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n" + "=" * 60)
    print("1. MISSING VALUE REPORT:")
    print("=" * 60)
    print(missing_report(df))
    print("\n[Filtered to only missing columns: only_missing=True]")
    print(missing_report(df, only_missing=True))

    print("\n" + "=" * 60)
    print("2. COLUMN SUMMARY:")
    print("=" * 60)
    print(column_summary(df))

    print("\n" + "=" * 60)
    print("3. DUPLICATE REPORT:")
    print("=" * 60)
    print(duplicate_report(df))
    print("\n[Extracted Duplicate Records (return_records=True)]:")
    print(duplicate_report(df, return_records=True))

    print("\n" + "=" * 60)
    print("4. UNIQUE & CARDINALITY REPORT:")
    print("=" * 60)
    print(unique_report(df))

    print("\n" + "=" * 60)
    print("5. NUMERIC SUMMARY:")
    print("=" * 60)
    print(numeric_summary(df))
    print("\n[Extended Diagnostics (extended=True)]:")
    print(numeric_summary(df, extended=True))

    print("\n" + "=" * 60)
    print("6. UNIFIED DATASET REPORT (pp.report(df)):")
    print("=" * 60)
    print(report(df))


if __name__ == "__main__":
    run_manual_test()