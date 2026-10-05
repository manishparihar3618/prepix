import pandas as pd

from prepix import (
    column_summary,
    duplicate_report,
    missing_report,
    unique_report,
)

data = {
    "Age": [21, None, 23, None, 20],
    "Salary": [50000, None, None, None, 45000],
    "City": ["Indore", None, "Delhi", "Delhi", "Mumbai"],
    "Gender": ["M", "F", "M", "F", "M"],
}

df = pd.DataFrame(data)
print("=== Missing Report ===")
print(missing_report(df))
print("\n=== Column Summary ===")
print(column_summary(df))
print("\n=== Duplicate Report ===")
print(duplicate_report(df))
print("\n=== Unique Report ===")
print(unique_report(df))