# Prepix

**Prepix** is a lightweight Python library for **data quality checks, preprocessing, and ML preparation**.

It aims to automate repetitive EDA and preprocessing tasks commonly performed before Machine Learning.

## 🚀 Current Features

- `missing_report()` — Analyze missing values and get basic preprocessing suggestions.
- `column_summary()` — Get a quick overview of column types, missing values, unique values, memory usage, and suggested types.
- `duplicate_report()` — Analyze duplicate rows, percentages, unique records, status, and recommendations.

## 📦 Installation

```bash
git clone https://github.com/manishparihar3618/prepix.git
cd prepix
pip install -e .
```

## 💻 Usage

```python
import prepix as pp
import pandas as pd

df = pd.read_csv("data.csv")

# Column overview
summary = pp.column_summary(df)
print(summary)

# Missing-value analysis
missing = pp.missing_report(df)
print(missing)

# Duplicate row analysis
duplicates = pp.duplicate_report(df)
print(duplicates)
```

## 📊 Example

### `column_summary()`

Returns information such as:

```text
Column | Data Type | Missing Values | Missing Percentage(%) | Unique Values | Memory(Bytes) | Suggested Type
Age    | float64   | 2              | 40.0                  | 3             | 40            | Numeric
City   | str       | 1              | 20.0                  | 3             | 282           | Categorical
```

### `missing_report()`

Returns:

```text
Column | Missing Values | Missing Percentage(%) | Status | Suggestion
Age    | 2              | 40.0                  | High   | Review Before Imputation
City   | 1              | 20.0                  | Medium | Consider Mode Imputation
```

### `duplicate_report()`

Returns:

```text
Total Rows | Duplicate Rows | Duplicate Percentage(%) | Unique Rows | Status          | Suggestion
1000       | 25             | 2.50                    | 975         | Duplicates Found| Consider reviewing and removing duplicate records
```

> **Note on duplicate counting:** A row is considered a duplicate if an identical row occurs earlier in the DataFrame (following pandas `df.duplicated()` semantics). The original DataFrame remains unmodified.

## 🎯 Project Goal

Prepix aims to automate common **EDA, data-preprocessing, and feature-engineering tasks** to make ML data preparation:

- Faster
- Simpler
- More consistent
- Reusable

The project is designed as a layer **on top of pandas and the Python ML ecosystem**, not as a replacement for them.

## 🌱 Open-Source Approach

Prepix follows an iterative approach:

**Build MVP → Release → Get User Feedback → Improve → Repeat**

Future versions may include:

- Outlier detection
- Data cleaning
- Encoding
- Scaling
- Feature engineering
- ML preprocessing utilities

## 📄 License

MIT License

---

**Prepix — Automate repetitive data-preparation tasks and focus on building better ML workflows.**