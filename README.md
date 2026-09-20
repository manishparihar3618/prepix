# Prepix

**Prepix** is a lightweight Python library for **data quality checks, preprocessing, and ML preparation**.

It aims to automate repetitive EDA and preprocessing tasks commonly performed before Machine Learning.

## 🚀 Current Features

- `missing_report()` — Analyze missing values and get basic preprocessing suggestions.
- `column_summary()` — Get a quick overview of column types, missing values, unique values, memory usage, and suggested types.

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
```

## 📊 Example

### `column_summary()`

Returns information such as:

```text
Column | Data Type | Missing Values | Unique Values | Suggested Type
Age    | float64   | 2              | 50            | Numeric
City   | object    | 1              | 10            | Categorical
```

### `missing_report()`

Returns:

```text
Column | Missing Values | Missing % | Status | Suggestion
Age    | 2              | 4.0%      | Low    | Median Imputation
City   | 1              | 2.0%      | Low    | Mode Imputation
```

## 🎯 Project Goal

Prepix aims to automate common **EDA, data-preprocessing, and feature-engineering tasks** to make ML data preparation:

- Faster
- Simpler
- More consistent
- Reusable

The project is designed as a layer **on top of pandas and the Python ML ecosystem**, not as a replacement for them.

## 🌱 Open-Source Approach

Prepix will follow an iterative approach:

**Build MVP → Release → Get User Feedback → Improve → Repeat**

Future versions may include:

- Duplicate detection
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