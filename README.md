# Prepix

A lightweight Python library for data preprocessing, quality checks, and ML utilities.

## Features

- ✅ missing_report()

## Example

```python
import prepix as pp
import pandas as pd

df = pd.read_csv("data.csv")

report = pp.missing_report(df)

print(report)