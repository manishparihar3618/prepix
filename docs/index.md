# Prepix Documentation

Welcome to the official documentation for **Prepix** — a lightweight, reliable Python library for **automated data quality assessment, exploratory data analysis (EDA), and data validation**.

Built directly on top of `pandas` and `numpy`, Prepix empowers data analysts, data scientists, and machine learning engineers to rapidly profile datasets, detect subtle data flaws, and validate schema integrity before training models or executing production pipelines.

---

## Why Prepix?

When preparing datasets for analysis or machine learning, practitioners frequently encounter:
- **Hidden missing values**: Inconsistent null representations, unexpected missingness distributions.
- **Unintended duplicate rows**: Accidental re-ingestion of batches or redundant key-value combinations.
- **Subtle schema drift**: Silent type coercion, missing required columns, or violated uniqueness constraints.
- **Cardinality traps**: High-cardinality features masquerading as regular categories, or constant columns with zero variance.
- **Skewed distributions & extreme outliers**: Unnoticed extremes corrupting parametric models.

Writing repetitive custom pandas inspection scripts for every project is slow, error-prone, and inconsistent. Prepix standardizes and automates these checks with a clean, readable API that yields actionable diagnostic reports in milliseconds.

---

## Core Design Principles

1. **DataFrame Immutability**:
   All Prepix functions are strictly read-only. Your input DataFrames are never modified in place, columns are never dropped or cast silently, and indices are never altered.
2. **Transparent, Heuristic Guidance**:
   Recommendations (such as imputation strategies or deduplication flags) are advisory suggestions. Prepix provides the diagnostic evidence and leaves strategic decisions in your hands.
3. **Engineered for Edge Cases**:
   Prepix has been verified against empty DataFrames (`0x0`, `0xN`, `Nx0`), infinite values (`np.inf`, `-np.inf`), duplicate column headers, and unhashable cell types (such as nested lists or dictionaries).
4. **Zero Heavyweight Dependencies**:
   Prepix relies exclusively on `pandas>=2.0.0` and `numpy>=1.24.0`. No heavy Java runtimes, large visual dependencies, or bloated web servers are required.

---

## Public Functions Overview

Prepix exposes 7 carefully audited public analysis functions:

| Function | Primary Purpose | Key Return Information |
| :--- | :--- | :--- |
| [`missing_report(df)`](functions.md#missing_report) | Missing-value analysis and severity categorization | Missing counts, percentages, status (Low/Medium/High), and imputation suggestions |
| [`column_summary(df)`](functions.md#column_summary) | Comprehensive column structural profiling | Dtypes, null counts, null %, unique counts, deep memory usage, suggested types |
| [`duplicate_report(df)`](functions.md#duplicate_report) | Detection and extraction of duplicate records | Total rows, duplicate rows, duplicate %, unique rows, or duplicated records |
| [`unique_report(df)`](functions.md#unique_report) | Cardinality and uniqueness classification | Unique counts, unique %, cardinality tiers (`Constant` to `Unique`), identifier flags |
| [`numeric_summary(df)`](functions.md#numeric_summary) | Statistical profiling of numerical features | 5-number summary, mean, std, range; optional IQR, skewness, kurtosis, zero/negative/outlier counts |
| [`schema_report(df, schema)`](functions.md#schema_report) | DataFrame contract and schema validation | Column presence, data types, nullability, uniqueness, and unexpected column checks |
| [`report(df)`](functions.md#report) | Unified, multi-section quality diagnosis | Combined health overview, prioritized issues table, actionable recommendations, and sub-reports |

---

## Navigation

- **[Installation Guide](installation.md)**: Compatibility requirements, local editable install, and PyPI instructions.
- **[Quickstart Guide](quickstart.md)**: 5-minute hands-on walkthrough with realistic data.
- **[Data Sources Guide](data_sources.md)**: Ingesting DataFrames from CSV, JSON, SQL, Parquet, and Excel.
- **[Function Reference](functions.md)**: Full API parameters, defaults, return columns, and code snippets.
- **[Examples](examples.md)**: Detailed synthetic workflow scenarios.
- **[Troubleshooting & FAQ](troubleshooting.md)**: Unique count behaviors, duplicate semantics, and troubleshooting.
