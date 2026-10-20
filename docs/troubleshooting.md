# Troubleshooting & FAQ

This guide answers common architectural and behavioral questions, explains subtleties in statistical semantics, and provides troubleshooting advice for edge cases.

---

## 1. Why does `column_summary()` report fewer unique values than `unique_report()`?

This is one of the most important design distinctions in Prepix:

- **`column_summary(df, dropna=True)`** defaults to `dropna=True`. This follows standard pandas convention (`Series.nunique(dropna=True)`), where missing values (`NaN` / `None`) are excluded from the count of distinct values.
- **`unique_report(df, dropna=False)`** defaults to `dropna=False`. For cardinality classification and identifier analysis, whether a column has nulls is a vital distinct category. Therefore, null is treated as a separate unique value category by default.

### Example Demonstrating the Difference:
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({"Category": ["A", "A", "B", None]})

print(pp.column_summary(df)[["Column", "Unique Values"]])
# Output: Unique Values = 2 ('A' and 'B')

print(pp.unique_report(df)[["Column", "Unique Values"]])
# Output: Unique Values = 3 ('A', 'B', and None)
```

### Behavior in `pp.report(df)`:
When you generate the unified report via `pp.report(df)`, the parameter `dropna_unique=True` is applied across **both** sub-reports. This guarantees that `report.columns` and `report.unique` show identical unique value counts. If you wish to include nulls in the unified report, call `pp.report(df, dropna_unique=False)`.

---

## 2. How are duplicate rows counted in `duplicate_report()`?

Prepix adheres strictly to pandas `df.duplicated()` semantics:

- **When `keep='first'` (default)**:
  The first occurrence of any row is considered the original valid row. Only subsequent repeated rows are marked as duplicates.
  *Example*: If a row is repeated 3 times, `Duplicate Rows = 2` and `Unique Rows = 1`.
- **When `keep=False`**:
  All occurrences of duplicate rows are marked as duplicates.
  *Example*: If a row is repeated 3 times, `Duplicate Rows = 3` and `Unique Rows = 0`.
- **When `return_records=True`**:
  Prepix returns a DataFrame containing only the rows identified as duplicates according to your `keep` policy.

```python
# To inspect the duplicate rows directly:
duplicates = pp.duplicate_report(df, return_records=True)
print(duplicates)
```

---

## 3. How does Prepix handle unhashable objects (e.g. lists or dicts in cells)?

In standard pandas, calling `.duplicated()` or `.nunique()` on columns containing Python `list`, `dict`, or `set` objects raises a `TypeError: unhashable type: 'list'`.

Prepix includes fallback handling for these columns:
- Values are converted to deterministic string representations (`repr()`) internally during uniqueness and duplicate evaluation.
- The original input DataFrame is never modified.

---

## 4. How does Prepix handle infinite values (`np.inf`, `-np.inf`)?

Infinite values are treated according to standard numerical mathematics:
- They are recognized as valid numerical values and are not converted to `NaN`.
- In `numeric_summary()`, `np.isinf` boundaries are handled safely so that IQR calculation and outlier fencing do not crash or produce runtime divide-by-zero warnings.

---

## 5. Why does memory usage in `column_summary` differ from `df.info()`?

By default, pandas `df.memory_usage()` reports the shallow pointer footprint for `object` and string columns, which significantly underestimates actual RAM consumption.

Prepix always uses `series.memory_usage(deep=True)`, which traverses Python string objects and nested references to provide an accurate estimate of actual RAM utilization.

---

## 6. How do I silence warnings in automated pipelines?

Prepix functions are designed to operate cleanly without emitting unwanted `RuntimeWarning` messages. If you wish to filter pandas or system-level warnings in your own scripts, use Python's built-in `warnings` module:

```python
import warnings
warnings.filterwarnings("ignore")
```
