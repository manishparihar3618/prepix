# Prepix

**Prepix** is a lightweight, high-performance Python library for **automated data quality assessment, exploratory data analysis (EDA), and machine learning data preparation**.

Built on top of pandas and numpy, Prepix automates repetitive analysis workflows through simple, predictable APIs while providing transparent, cautious recommendations.

---

## 🚀 Core Features

- **`report(df)`** — **Unified High-Level Report**: Instant dataset health diagnosis combining dimensions, memory, missingness, duplicates, uniqueness, distributions, and prioritized actionable findings.
- **`column_summary(df)`** — **Structural Column Overview**: Per-column data types, missing counts, uniqueness, deep memory footprint, and heuristic suggested types.
- **`missing_report(df, ...)`** — **Missing-Value Audit**: Configurable severity thresholds, optional filtering of clean columns (`only_missing=True`), and conservative imputation suggestions.
- **`duplicate_report(df, ...)`** — **Duplicate Record Analysis**: Full-row or column-subset duplicate audits (`subset=[...]`), duplicate percentages, and direct duplicate record inspection (`return_records=True`).
- **`unique_report(df, ...)`** — **Uniqueness & Cardinality**: Cardinality classifications, constant-column detection, and distinction between continuous numeric variables and categorical/string identifiers.
- **`numeric_summary(df, ...)`** — **Descriptive Statistics**: Comprehensive 5-number summaries, dispersion, range, and optional extended distribution diagnostics (`extended=True`: IQR, skewness, kurtosis, zero/negative counts, and 1.5x IQR Tukey outlier detection).

---

## 📦 Installation

```bash
git clone https://github.com/manishparihar3618/prepix.git
cd prepix
pip install -e .
```

---

## 💻 Quickstart

### 1. Unified Dataset Report

```python
import pandas as pd
import prepix as pp

df = pd.read_csv("data.csv")

# Generate the complete dataset report
rep = pp.report(df)

# Print a formatted diagnostic summary
print(rep)

# Or access specialized sub-reports individually:
print(rep.overview)    # High-level dataset metrics
print(rep.issues)      # Prioritized findings and warnings
print(rep.missing)     # Missing values table
print(rep.numeric)     # Numerical feature statistics
print(rep.unique)      # Uniqueness and cardinality
print(rep.duplicates)  # Duplicate analysis
print(rep.columns)     # Column metadata and memory

# Export report sections to JSON or dict:
rep.to_json("report.json", indent=2)
report_dict = rep.to_dict()
```

### 2. Specialized Analysis Functions

#### Column Summary
```python
summary = pp.column_summary(df)
print(summary)
```
*Columns returned:* `Column`, `Data Type`, `Missing Values`, `Missing Percentage(%)`, `Unique Values`, `Memory(Bytes)`, `Suggested Type`.

#### Missing Values Report
```python
# Standard report sorted by missing percentage descending
missing = pp.missing_report(df, low_threshold=10.0, medium_threshold=30.0)

# In wide datasets, focus only on columns with missing values:
only_missing = pp.missing_report(df, only_missing=True)
print(only_missing)
```
*Columns returned:* `Column`, `Missing Values`, `Missing Percentage(%)`, `Status`, `Suggestion`.

#### Duplicate Rows Report
```python
# Full-row duplicate summary
dup_summary = pp.duplicate_report(df)
print(dup_summary)

# Check duplicates on specific key columns:
dup_keys = pp.duplicate_report(df, subset=["customer_id", "order_date"])

# Extract duplicate records for manual inspection:
dup_records = pp.duplicate_report(df, return_records=True)
print(dup_records.head())
```

#### Uniqueness & Cardinality
```python
# Uniqueness analysis (NaN counted as distinct category by default)
unique_info = pp.unique_report(df, dropna=False)

# Exclude NaN from unique counts:
unique_no_nulls = pp.unique_report(df, dropna=True)
print(unique_info)
```
*Cardinality levels:* `Constant`, `Low`, `Medium`, `High`, `Very High`, `Very High (Unique)`.

#### Numeric Distribution Summary
```python
# Standard 12-statistic summary:
num_stats = pp.numeric_summary(df)

# Extended diagnostics: IQR, Skewness, Kurtosis, Zeros, Negatives, Outliers:
num_extended = pp.numeric_summary(df, extended=True)

# Custom percentiles (e.g. 5th and 95th percentiles):
num_percentiles = pp.numeric_summary(df, percentiles=[0.05, 0.95])
print(num_extended)
```

---

## 📊 Sample Unified Report Output

```text
==============================================================
                    PREPIX QUALITY REPORT                    
==============================================================

1. DATASET OVERVIEW
--------------------------------------------------------------
  • Rows                        : 1,000
  • Columns                     : 10
  • Total Cells                 : 10,000
  • Memory Usage                : 142.50 KB
  • Numeric Columns             : 5
  • Categorical Columns         : 3
  • Datetime Columns            : 1
  • Boolean Columns             : 1
  • Other Columns               : 0
  • Total Missing Cells         : 150 (1.50%)
  • Columns with Missing        : 2
  • Duplicate Rows              : 12 (1.20%)

2. KEY FINDINGS & ISSUES
--------------------------------------------------------------
  • [MEDIUM] Dataset: Duplicate Records — 12 duplicate rows (1.20%) detected
  • [HIGH  ] Column 'Salary': High Missingness — 450 missing values (45.00%)
  • [LOW   ] Column 'IsActive': Zero Variance (Constant) — Contains single constant value
  • [LOW   ] Column 'CustomerID': Potential Identifier — 100% unique values (1000/1000)
  • [LOW   ] Column 'Income': Potential Outliers — 14 values outside 1.5x IQR boundaries

3. RECOMMENDED ACTIONS
--------------------------------------------------------------
  • Review and consider deduplicating using df.drop_duplicates()
  • Investigate cause of missingness in 'Salary'; review before imputation or drop
  • Consider dropping constant column 'IsActive' prior to ML modeling
  • Verify whether 'CustomerID' is a primary key/ID; exclude from predictive features
  • Inspect distribution and consider robust scaling or clipping for 'Income'
==============================================================
```

---

## 🛡️ Design Principles & Integrity

1. **DataFrame Immutability**: All Prepix operations are strictly read-only; input DataFrames are never mutated, re-indexed, or modified in place.
2. **Transparent Recommendations**: Suggestions are heuristic guidelines, not automated decisions. Prepix does not silently drop, impute, or cast columns.
3. **Robust Handling of Edge Cases**:
   - Empty DataFrames (0x0, 0xN, Nx0)
   - Datasets with duplicate column headers or duplicate indexes
   - Infinite values (`np.inf`, `-np.inf`) without runtime crashes or warnings
   - Unhashable objects (lists, dictionaries) in cell values
   - Wide (hundreds of columns) and tall (hundreds of thousands of rows) datasets

---

## 🧪 Running Tests & Benchmarks

Run the test suite using Python's built-in `unittest`:

```bash
python -m unittest discover -s test -p "test_*.py" -v
```

Run performance benchmarks:

```bash
python benchmark.py
```

---

## 📄 License

MIT License. Copyright (c) Manish Parihar.