# Quickstart Tutorial

Get up and running with Prepix in under 5 minutes. This tutorial demonstrates how to load a representative dataset, generate an instant health report, and inspect specific data quality vectors.

---

## 1. Import Prepix and Load Data

Prepix is conventionally imported with the shorthand alias `pp`:

```python
import pandas as pd
import prepix as pp

# Create a sample customer orders dataset
df = pd.DataFrame({
    "OrderID": [1001, 1002, 1003, 1004, 1005, 1005],  # Contains duplicate 1005
    "Customer": ["Alice", "Bob", "Charlie", "David", "Emma", "Emma"],
    "Age": [28.0, 34.0, None, 45.0, 29.0, 29.0],       # Contains 1 missing value
    "OrderTotal": [150.50, 89.00, 210.00, 45.00, 999.00, 999.00],
    "StoreCountry": ["USA", "USA", "USA", "USA", "USA", "USA"],  # Constant column
})
```

---

## 2. Generate a Complete Quality Report

The fastest way to understand your dataset is with `pp.report(df)`:

```python
report = pp.report(df)

# Print a human-readable CLI summary
print(report)
```

The output gives you an instant overview:
1. **Dataset Overview**: Dimensions (6 rows, 5 columns), memory usage, count of missing values, and duplicate row percentage.
2. **Key Findings & Issues**: Highlighted severity warnings:
   - High/Medium missingness
   - Duplicate rows detected
   - Constant columns (zero variance)
   - High-cardinality features or potential primary keys
   - Extreme distribution outliers
3. **Recommended Actions**: Clear, actionable guidance on next steps.

---

## 3. Accessing Specialized Sub-Reports

The returned `PrepixReport` object provides direct access to granular sub-reports as standard pandas DataFrames:

```python
# 1. Dataset-level overview metrics
print(report.overview)

# 2. Prioritized quality issues and findings
print(report.issues)

# 3. Column structural metadata
print(report.columns)

# 4. Detailed missing values and suggestions
print(report.missing)

# 5. Uniqueness and cardinality classifications
print(report.unique)

# 6. Numerical distributions and 5-number statistics
print(report.numeric)

# 7. Duplicate row metrics
print(report.duplicates)
```

---

## 4. Exporting Reports

Export the unified analysis directly to JSON for automated CI/CD pipelines, logging, or web dashboards:

```python
# Export to a JSON file
report.to_json("data_quality_report.json", indent=2)

# Or convert to a standard Python dictionary
data_dict = report.to_dict()
print(f"Detected issues count: {len(data_dict['issues'])}")
```

---

## 5. Running Individual Reports

When focusing on a specific analysis task, invoke individual functions directly:

```python
# Check missing values
missing = pp.missing_report(df, only_missing=True)

# Profile column types and memory
summary = pp.column_summary(df)

# Check and extract duplicate records
dups = pp.duplicate_report(df, return_records=True)

# Audit numerical distributions with extended skew/outliers
num_stats = pp.numeric_summary(df, extended=True)

# Validate against expected schema
schema = {
    "OrderID": {"type": "integer", "nullable": False, "unique": True},
    "Customer": "string",
    "Age": {"type": "numeric", "nullable": False},
}
validation = pp.schema_report(df, schema=schema, allow_unexpected=True)
print(validation)
```

Next, see the [Function Reference](functions.md) for full parameter specifications and options.
