# Prepix

**Prepix** is a lightweight, reliable Python library for **automated data quality assessment, exploratory data analysis (EDA), and data contract validation**, built on top of pandas and numpy.

[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/manishparihar3618/prepix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/Status-v0.1.0%20Beta-orange)](CHANGELOG.md)

---

## 📌 The Problem Prepix Solves

Data scientists, analysts, and ML engineers spend up to 80% of their time cleaning and validating tabular data. Common pitfalls include:
- **Silent missingness** and inconsistent null values across critical feature columns.
- **Accidental duplicate rows** introduced during data ingestion or joins.
- **Silent schema drift**, such as missing required columns or unexpected type coercion.
- **Cardinality traps**, where constant columns provide zero variance or high-cardinality strings are inadvertently one-hot encoded.
- **Extreme outliers and severe distribution skew** that corrupt statistical estimates and model training.

Writing ad-hoc pandas queries to check each of these issues for every new dataset is repetitive, error-prone, and difficult to standardize across a team.

**Prepix automates these inspections through simple, composable, and immutability-guaranteed functions**, producing structured diagnostic DataFrames and instant human-readable health reports.

---

## 🎯 Intended Users & Use Cases

- **Data Analysts**: Quickly audit incoming CSV/Excel files and export structured quality summaries.
- **Machine Learning Engineers**: Validate training and inference feature sets against expected data contracts to prevent silent pipeline failures.
- **Data Engineers**: Embed automated data validation checks into ETL jobs and CI/CD testing suites.
- **Researchers**: Inspect distribution characteristics, skewness, kurtosis, and Tukey outliers across experimental datasets.

---

## 📦 Installation

Prepix requires **Python >= 3.9**, **pandas >= 2.0.0**, and **numpy >= 1.24.0**.

### Development / Local Installation
Until the package is officially published to PyPI, install Prepix directly from the repository in editable mode:

```bash
git clone https://github.com/manishparihar3618/prepix.git
cd prepix
pip install -e .
```

### PyPI Installation (Post-Release)
Once published:
```bash
pip install prepix
```

---

## ⚡ Quickstart

```python
import pandas as pd
import prepix as pp

# Load or define your dataset
df = pd.DataFrame({
    "OrderID": [101, 102, 103, 104, 104],
    "Customer": ["Alice", "Bob", "Charlie", "David", "David"],
    "Age": [28, None, 35, 42, 42],
    "Revenue": [150.0, 220.5, 310.0, 95.0, 95.0],
    "Region": ["North", "North", "North", "North", "North"],
})

# Generate the complete dataset quality report
rep = pp.report(df)

# Print a formatted diagnostic summary to console
print(rep)
```

### Output Summary
```text
==============================================================
                    PREPIX QUALITY REPORT                    
==============================================================

1. DATASET OVERVIEW
--------------------------------------------------------------
  - Rows                        : 5
  - Columns                     : 5
  - Total Cells                 : 25
  - Memory Usage                : 1.24 KB
  - Numeric Columns             : 3
  - Categorical Columns         : 2
  - Datetime Columns            : 0
  - Boolean Columns             : 0
  - Other Columns               : 0
  - Total Missing Cells         : 1 (4.00%)
  - Columns with Missing        : 1
  - Duplicate Rows              : 1 (20.00%)

2. KEY FINDINGS & ISSUES
--------------------------------------------------------------
  [MEDIUM] Dataset: Duplicate Records - 1 duplicate rows (20.00%) detected
  [MEDIUM] Column 'Age': Moderate Missingness - 1 missing values (20.00%)
  [LOW   ] Column 'Region': Zero Variance (Constant) - Contains single constant value

3. RECOMMENDED ACTIONS
--------------------------------------------------------------
  * Review and consider deduplicating using df.drop_duplicates()
  * Consider median/mode imputation or indicator flag for 'Age'
  * Consider dropping constant column 'Region' prior to ML modeling
==============================================================
```

---

## 🔍 Verified Public Functions

| Function | Primary Purpose | Key Return Data |
| :--- | :--- | :--- |
| **`report(df)`** | Unified multi-section quality analysis | Returns `PrepixReport` with overview, issues, and all sub-reports |
| **`missing_report(df)`** | Missing-value audit & imputation recommendations | `Column`, `Missing Values`, `Missing Percentage(%)`, `Status`, `Suggestion` |
| **`column_summary(df)`** | Structural column profiling & memory estimation | `Column`, `Data Type`, `Missing Values`, `Unique Values`, `Memory(Bytes)`, `Suggested Type` |
| **`duplicate_report(df)`** | Duplicate record identification & extraction | `Total Rows`, `Duplicate Rows`, `Duplicate %`, `Unique Rows`, `Status`, `Suggestion` (or duplicate rows via `return_records=True`) |
| **`unique_report(df)`** | Cardinality categorization & identifier detection | `Column`, `Unique Values`, `Cardinality`, `Status`, `Suggestion` |
| **`numeric_summary(df)`** | Statistical summary & distribution diagnostics | 5-number summary, mean, std, range; plus IQR, skewness, kurtosis, zero/negative/outlier counts with `extended=True` |
| **`schema_report(df, schema)`** | Schema validation & contract verification | `Column`, `Check`, `Expected`, `Actual`, `Status`, `Message` |

---

## 🛠️ Specialized Function Examples

### 1. Missing Values
```python
# Report columns sorted by missingness percentage
missing = pp.missing_report(df, only_missing=True)
print(missing)
```

### 2. Duplicate Detection & Extraction
```python
# Summary statistics
print(pp.duplicate_report(df))

# Extract the duplicate rows directly for inspection
dups = pp.duplicate_report(df, return_records=True)
print(dups)
```

### 3. Numerical Diagnostics
```python
# Extended statistics: IQR, skewness, kurtosis, zeros, negatives, and Tukey outliers
num_stats = pp.numeric_summary(df, extended=True, percentiles=[0.05, 0.95])
print(num_stats)
```

### 4. Schema Contract Validation
```python
schema = {
    "OrderID": {"type": "integer", "nullable": False, "unique": True},
    "Customer": "string",
    "Age": {"type": "numeric", "nullable": False},
}
val_report = pp.schema_report(df, schema=schema, allow_unexpected=True)
print(val_report)
```

---

## 🛡️ Core Guarantees

1. **DataFrame Immutability**: All Prepix operations are strictly read-only; input DataFrames are never modified in place.
2. **Transparent Recommendations**: Suggestions are heuristic guidelines, not automated decisions. Prepix does not silently drop, impute, or cast columns.
3. **Resilient Edge-Case Handling**: Tested against empty DataFrames (`0x0`, `0xN`, `Nx0`), infinite values (`np.inf`, `-np.inf`), duplicate column names, and unhashable cell types.
4. **Minimal Dependencies**: Strictly depends only on `pandas>=2.0.0` and `numpy>=1.24.0`.

---

## 🧪 Testing & Verification

Run the full automated test suite using Python's built-in `unittest`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Execute all verified synthetic workflow examples:

```bash
python examples/run_all_examples.py
```

Run performance benchmarks:

```bash
python benchmark.py
```

---

## 📖 Documentation Links

- [Documentation Index](docs/index.md)
- [Installation Guide](docs/installation.md)
- [Quickstart Guide](docs/quickstart.md)
- [Data Sources Guide](docs/data_sources.md)
- [Function Reference](docs/functions.md)
- [Examples Guide](docs/examples.md)
- [Troubleshooting & FAQ](docs/troubleshooting.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)

---

## 📢 Release Status

Prepix is currently at **v0.1.0 (Beta)**. Core functionality is complete and verified with 145 unit tests across diverse data scales and types. We are preparing for the first official PyPI release.

If you encounter any issues or have feedback, please open an issue on the [GitHub Issue Tracker](https://github.com/manishparihar3618/prepix/issues).

---

## 📄 License

Prepix is licensed under the [MIT License](LICENSE). Copyright (c) Manish Parihar.