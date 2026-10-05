import pandas as pd

from prepix import column_summary, duplicate_report, missing_report

data = {
    "Age": [21, None, 23, None, 20],
    "Salary": [50000, None, None, None, 45000],
    "City": ["Indore", None, "Delhi", "Delhi", "Mumbai"],
    "Gender": ["M", "F", "M", "F", "M"],
}

df = pd.DataFrame(data)
print(missing_report(df))
print(column_summary(df))
print(duplicate_report(df))