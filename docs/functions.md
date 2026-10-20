# Function Reference

This document provides a verified, complete reference for all public functions and classes in **Prepix**.

---

## Table of Contents
1. [`report()`](#report) — Unified Dataset Report
2. [`missing_report()`](#missing_report) — Missing-Value Analysis
3. [`column_summary()`](#column_summary) — Column Structural Profiling
4. [`duplicate_report()`](#duplicate_report) — Duplicate Row Detection
5. [`unique_report()`](#unique_report) — Uniqueness & Cardinality
6. [`numeric_summary()`](#numeric_summary) — Numerical Statistics
7. [`schema_report()`](#schema_report) — Schema & Contract Validation
8. [`PrepixReport`](#prepixreport) — Report Container Class

---

## `report()`

### Purpose
Generates a unified data quality and exploratory data analysis (EDA) report. Combines dataset dimensions, memory footprint, missing-value distribution, duplicate records, column cardinality, numerical 5-number summaries, distribution diagnostics, prioritized issues, and actionable recommendations into a single `PrepixReport` object.

### Signature
```python
prepix.report(
    df: pd.DataFrame,
    low_missing_threshold: float = 10.0,
    medium_missing_threshold: float = 30.0,
    numeric_extended: bool = True,
    dropna_unique: bool = True,
) -> prepix.PrepixReport
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame to analyze. Never modified.
- **`low_missing_threshold`** (*float*, default `10.0`): Percentage threshold up to which missingness is classified as "Low".
- **`medium_missing_threshold`** (*float*, default `30.0`): Percentage threshold up to which missingness is classified as "Medium". Values exceeding this are marked "High".
- **`numeric_extended`** (*bool*, default `True`): If True, computes extended numeric diagnostics (IQR, Skewness, Kurtosis, Zeros, Negatives, Outliers).
- **`dropna_unique`** (*bool*, default `True`): Null-handling policy for unique counts. When `True` (default), null values are excluded from unique counts in both the `columns` and `unique` sub-reports, guaranteeing that unique counts remain fully consistent across all sections of the report.

### Returns
- **`PrepixReport`**: Object containing `.overview`, `.issues`, `.columns`, `.missing`, `.unique`, `.numeric`, `.duplicates`, with helper methods `.to_dict()` and `.to_json()`.

### Key Output Columns (in sub-reports)
- **`report.overview`**: `Metric`, `Value` (Rows, Columns, Cells, Memory, Types breakdown, Missing Cells, Duplicate Rows).
- **`report.issues`**: `Severity` (`HIGH`, `MEDIUM`, `LOW`, `INFO`), `Column`, `Issue`, `Detail`, `Recommendation`.

### Edge Cases & Exceptions
- Raises `TypeError` if `df` is not a `pd.DataFrame`.
- Gracefully handles empty DataFrames (`0x0`, `0xN`, `Nx0`) with zero crashes.

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "A": [1, 2, 3, 4, None],
    "B": ["x", "y", "x", "y", "x"],
    "C": [10.5, 20.2, 10.5, 30.1, 40.0],
})

rep = pp.report(df)
print(rep)
```

---

## `missing_report()`

### Purpose
Calculates missing-value counts, percentages, and classifications across all columns, sorted in descending order of missingness. Provides heuristic suggestions for data imputation.

### Signature
```python
prepix.missing_report(
    df: pd.DataFrame,
    low_threshold: float = 10.0,
    medium_threshold: float = 30.0,
    only_missing: bool = False,
) -> pd.DataFrame
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame.
- **`low_threshold`** (*float*, default `10.0`): Percentage cutoff for "Low" missingness. Must be between 0 and 100.
- **`medium_threshold`** (*float*, default `30.0`): Percentage cutoff for "Medium" missingness. Must be between 0 and 100, and `>= low_threshold`.
- **`only_missing`** (*bool*, default `False`): If `True`, filters out columns with 0 missing values, returning only columns with missing data.

### Return Structure (`pd.DataFrame`)
| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Column` | `object` | Name of the column |
| `Missing Values` | `int64` | Total count of missing/NaN values |
| `Missing Percentage(%)` | `float64` | Percentage of total rows with missing values (rounded to 2 decimal places) |
| `Status` | `object` | Category: `No Missing`, `Low`, `Medium`, or `High` |
| `Suggestion` | `object` | Recommended action: `No Action Needed`, `Consider Median Imputation`, `Consider Mode Imputation`, or `Review Before Imputation` |

### Edge Cases & Validation
- Raises `TypeError` if thresholds are not numeric or `only_missing` is not boolean.
- Raises `ValueError` if thresholds are `< 0`, `> 100`, NaN, or if `low_threshold > medium_threshold`.
- Supports DataFrames with duplicate column headers using positional indexing.

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "Age": [25, None, 30, None, 40],
    "Score": [88.5, 92.0, None, 75.0, 99.0],
    "City": ["NYC", "LA", "NYC", "Chicago", "LA"],
})

# Filter only columns containing nulls
missing = pp.missing_report(df, only_missing=True)
print(missing)
```

---

## `column_summary()`

### Purpose
Generates a structural overview of every column in the dataset, including data types, null statistics, cardinality, deep memory consumption, and suggested semantic types.

### Signature
```python
prepix.column_summary(
    df: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame.
- **`dropna`** (*bool*, default `True`): Whether to exclude NaN values when computing unique counts. When `True` (default), follows pandas `Series.nunique()` standard behavior. When `False`, nulls count as a distinct category.

### Return Structure (`pd.DataFrame`)
| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Column` | `object` | Column name |
| `Data Type` | `object` | Native pandas dtype string (e.g. `int64`, `float64`, `object`) |
| `Missing Values` | `int64` | Number of missing values |
| `Missing Percentage(%)` | `float64` | Percentage of missing values |
| `Unique Values` | `int64` | Number of distinct values (subject to `dropna`) |
| `Memory(Bytes)` | `int64` | Deep memory usage in bytes |
| `Suggested Type` | `object` | Semantic classification: `Binary`, `Numeric`, `Categorical`, `Categorical (2 values)`, `Datetime`, `Timedelta`, `Period` |

### Important Semantics Note
Notice that `column_summary()` defaults to `dropna=True`. If a column has 10 rows with values `[1, 1, 2, None]`, `column_summary` reports `Unique Values: 2`. By contrast, `unique_report()` defaults to `dropna=False` (which reports `3`). In `pp.report(df)`, `dropna_unique=True` is applied to both for guaranteed consistency.

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "IsActive": [True, False, True],
    "Amount": [10.50, 20.00, 30.25],
    "Label": ["A", "B", "A"],
})
print(pp.column_summary(df))
```

---

## `duplicate_report()`

### Purpose
Audits row duplicates across the entire DataFrame or a designated subset of columns, providing counts, percentages, and deduplication guidance, or returning the duplicate rows directly.

### Signature
```python
prepix.duplicate_report(
    df: pd.DataFrame,
    subset: Optional[Union[Sequence[str], str]] = None,
    keep: Union[str, bool] = "first",
    return_records: bool = False,
) -> pd.DataFrame
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame.
- **`subset`** (*str* or *sequence of str*, optional): Column name(s) to consider when identifying duplicates. If `None` (default), uses all columns.
- **`keep`** (*{'first', 'last', False}*, default `'first'`):
  - `'first'`: Marks subsequent duplicate rows as duplicates, preserving the first occurrence.
  - `'last'`: Marks preceding duplicate rows as duplicates, preserving the last occurrence.
  - `False`: Marks all occurrences of duplicate rows as duplicates.
- **`return_records`** (*bool*, default `False`):
  - If `False` (default): Returns the single-row duplicate statistics summary.
  - If `True`: Returns a DataFrame containing the actual duplicate records matching `keep`.

### Return Structure (when `return_records=False`)
| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Total Rows` | `int64` | Total row count in DataFrame |
| `Duplicate Rows` | `int64` | Count of rows flagged as duplicate |
| `Duplicate Percentage(%)` | `float64` | Percentage of duplicate rows |
| `Unique Rows` | `int64` | Total Rows minus Duplicate Rows |
| `Status` | `object` | `No Duplicates` or `Duplicates Found` |
| `Suggestion` | `object` | Actionable recommendation |

### Duplicate Row Semantics Explained
With `keep='first'` (default), if a row `{"ID": 1, "Val": "A"}` occurs 3 times:
- Row 1: Treated as the original unique row.
- Rows 2 & 3: Counted as duplicates (Duplicate Rows = 2, Unique Rows = 1).
If `keep=False`, all 3 rows are flagged as duplicates (Duplicate Rows = 3, Unique Rows = 0).

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "UserID": [1, 2, 2, 3, 3, 3],
    "Score": [10, 20, 20, 30, 30, 30],
})

# Summary
print(pp.duplicate_report(df))

# Extract the duplicate rows directly
dup_rows = pp.duplicate_report(df, return_records=True)
print(dup_rows)
```

---

## `unique_report()`

### Purpose
Audits cardinality and uniqueness for every column, categorizing columns into variance levels (`Constant` to `Unique`) and detecting potential primary keys or unique identifiers.

### Signature
```python
prepix.unique_report(
    df: pd.DataFrame,
    dropna: bool = False,
) -> pd.DataFrame
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame.
- **`dropna`** (*bool*, default `False`): Whether to exclude NaN values when computing unique counts. When `False` (default), null values are counted as a distinct category. Set `dropna=True` to ignore nulls.

### Return Structure (`pd.DataFrame`)
| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Column` | `object` | Name of the column |
| `Data Type` | `object` | Column data type |
| `Total Values` | `int64` | Total number of rows |
| `Unique Values` | `int64` | Number of distinct values |
| `Unique Percentage(%)` | `float64` | Unique values divided by Total Values * 100 |
| `Duplicate Values` | `int64` | Total Values minus Unique Values |
| `Cardinality` | `object` | Cardinality tier: `Constant`, `Low`, `Medium`, `High`, `Very High`, `Very High (Unique)` |
| `Status` | `object` | Diagnostic status: `Zero Variance`, `Binary`, `Low Diversity`, `Moderate Diversity`, `High Diversity`, `Near-Unique`, `Potential Identifier` |
| `Suggestion` | `object` | Recommendation for feature selection or encoding |

### Cardinality Tiers
- **`Constant`**: Exactly 1 unique value (Zero variance).
- **`Very High (Unique)`**: 100% unique values (Potential primary key / ID).
- **`Very High`**: `>= 90%` unique values.
- **`High`**: `50% - 90%` unique values.
- **`Medium`**: `10% - 50%` unique values.
- **`Low`**: `< 10%` unique values.

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "UUID": ["u1", "u2", "u3", "u4"],
    "Status": ["active", "active", "active", "active"],
    "Tier": ["silver", "gold", "silver", "bronze"],
})
print(pp.unique_report(df))
```

---

## `numeric_summary()`

### Purpose
Calculates detailed descriptive statistics and distribution metrics for all numerical columns. Excludes boolean and complex columns.

### Signature
```python
prepix.numeric_summary(
    df: pd.DataFrame,
    extended: bool = False,
    percentiles: Optional[Sequence[float]] = None,
) -> pd.DataFrame
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame.
- **`extended`** (*bool*, default `False`): If `True`, includes advanced distribution diagnostics: IQR, Skewness, Kurtosis, Zeros, Negatives, and Outliers.
- **`percentiles`** (*sequence of float*, optional): Custom percentiles to compute (e.g. `[0.05, 0.95]`). Values must be floats between `0.0` and `1.0`.

### Return Structure (`pd.DataFrame`)
- **Standard Columns**:
  - `Column`, `Data Type`, `Count` (valid non-nulls), `Missing Values`, `Mean`, `Median`, `Std` (sample standard deviation, ddof=1), `Min`, `Q1` (25th percentile), `Q3` (75th percentile), `Max`, `Range` (`Max - Min`).
- **Extended Columns** (when `extended=True`):
  - `IQR`: Interquartile Range (`Q3 - Q1`).
  - `Skewness`: Fisher-Pearson distribution asymmetry.
  - `Kurtosis`: Fisher excess peakedness.
  - `Zeros`: Count of values equal to 0.
  - `Negatives`: Count of values strictly less than 0.
  - `Outliers`: Count of values beyond the Tukey fence `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
- **Percentile Columns**: Added dynamically based on `percentiles` parameter (e.g. `P5`, `P95`).

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "Age": [22, 25, 29, 35, 80],
    "Salary": [45000, 52000, 61000, 75000, 150000],
    "Bonus": [0, 500, 1000, 0, 10000],
})

# Standard summary
print(pp.numeric_summary(df))

# Extended diagnostics with custom percentiles
print(pp.numeric_summary(df, extended=True, percentiles=[0.01, 0.99]))
```

---

## `schema_report()`

### Purpose
Validates a DataFrame against a contract schema, verifying required column presence, data types, nullability restrictions, uniqueness constraints, and unexpected columns without modifying the data.

### Signature
```python
prepix.schema_report(
    df: pd.DataFrame,
    schema: Dict[str, Union[str, type, Dict[str, Any]]],
    allow_unexpected: bool = True,
) -> pd.DataFrame
```

### Parameters
- **`df`** (*pandas.DataFrame*): Input DataFrame to validate.
- **`schema`** (*dict*): Mapping of column names to expected schema specifications. Specifications can be:
  - Shorthand string: `'string'`, `'numeric'`, `'integer'`, `'float'`, `'boolean'`, `'datetime'`, `'timedelta'`, `'category'`, `'any'`.
  - Python or NumPy type: `str`, `int`, `float`, `bool`, `datetime.date`, etc.
  - Specification dictionary with keys:
    - `'type'`: Expected data type (str or type).
    - `'nullable'`: `bool` (default `True`), whether nulls are allowed.
    - `'non_null'`: `bool`, alias for `nullable=False`.
    - `'unique'`: `bool` (default `False`), whether all values must be unique.
    - `'required'`: `bool` (default `True`), whether the column must exist in `df`.
- **`allow_unexpected`** (*bool*, default `True`): If `False`, columns in `df` not declared in `schema` are flagged with status `'FAIL'`.

### Return Structure (`pd.DataFrame`)
| Column | Type | Meaning |
| :--- | :--- | :--- |
| `Column` | `object` | Column name tested |
| `Check` | `object` | Validation check: `column_presence`, `data_type`, `non_null`, `unique`, `unexpected_column` |
| `Expected` | `object` | Expected contract condition |
| `Actual` | `object` | Observed value/condition in DataFrame |
| `Status` | `object` | `PASS` or `FAIL` |
| `Message` | `object` | Clear explanation of the check result |

### Example
```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "UserID": [101, 102, 103],
    "Email": ["a@test.com", "b@test.com", "c@test.com"],
    "Score": [95.0, None, 88.0],
    "ExtraCol": [1, 2, 3],
})

schema = {
    "UserID": {"type": "integer", "nullable": False, "unique": True},
    "Email": "string",
    "Score": {"type": "numeric", "nullable": False},  # Will FAIL due to None
}

results = pp.schema_report(df, schema=schema, allow_unexpected=False)
print(results)
```

---

## `PrepixReport`

### Purpose
Returned by `pp.report(df)`. Serves as a structured, inspectable container holding all sub-reports and metadata.

### Attributes
- **`.overview`** (*pd.DataFrame*): Dataset-level metrics (shape, memory, null percentage, duplicate count).
- **`.issues`** (*pd.DataFrame*): Prioritized data quality findings categorized by severity.
- **`.columns`** (*pd.DataFrame*): Column-by-column structural profile (`column_summary`).
- **`.missing`** (*pd.DataFrame*): Missing-value table and imputation suggestions (`missing_report`).
- **`.unique`** (*pd.DataFrame*): Cardinality and identifier analysis (`unique_report`).
- **`.numeric`** (*pd.DataFrame*): Numerical descriptive statistics (`numeric_summary`).
- **`.duplicates`** (*pd.DataFrame*): Duplicate row summary (`duplicate_report`).

### Methods
- **`to_dict() -> Dict[str, Any]`**: Converts all sub-reports to a nested dictionary representation.
- **`to_json(path_or_buf: Optional[str] = None, indent: int = 2) -> Optional[str]`**: Serializes all sub-reports to formatted JSON. If `path_or_buf` is provided, writes directly to file; otherwise, returns the JSON string.
- **`__str__()` / `__repr__()`**: Returns a formatted ASCII terminal report with sections for overview, key findings, and recommended actions.
