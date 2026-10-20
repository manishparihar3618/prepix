# Changelog

All notable changes to the Prepix project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-19

### Added
- **Unified Quality Report (`report`)**:
  - High-level dataset health diagnostics combining shape, memory, missingness, duplicate rows, cardinality, and numerical distributions into a `PrepixReport` object.
  - Prioritized findings table categorized by severity (`HIGH`, `MEDIUM`, `LOW`, `INFO`).
  - Actionable data preparation and feature engineering suggestions.
  - Serialization to Python dictionary (`to_dict()`) and JSON (`to_json()`).
- **Missing Value Report (`missing_report`)**:
  - Configurable missingness severity thresholds (`low_threshold`, `medium_threshold`).
  - Column filtering (`only_missing=True`) for wide datasets.
  - Heuristic imputation strategies based on column data type (median for numeric, mode for categorical/boolean).
- **Column Profiling (`column_summary`)**:
  - Column data types, missing counts, missing percentage, unique counts, and deep memory consumption.
  - Semantic suggested data types (`Binary`, `Numeric`, `Categorical`, `Datetime`, `Timedelta`, `Period`).
  - Configurable null-handling via `dropna` parameter.
- **Duplicate Record Detection (`duplicate_report`)**:
  - Row-level and subset-level duplicate detection (`subset=[...]`).
  - Flexible duplicate identification logic via `keep` (`'first'`, `'last'`, `False`).
  - Direct retrieval of duplicate rows via `return_records=True`.
- **Uniqueness & Cardinality Profiling (`unique_report`)**:
  - Six-tier cardinality classification: `Constant`, `Low`, `Medium`, `High`, `Very High`, `Very High (Unique)`.
  - Zero-variance detection and primary-key / unique identifier flagging.
  - Distinguishes continuous float variables from discrete categorical identifiers.
- **Numerical Feature Profiling (`numeric_summary`)**:
  - Standard five-number summary: count, missing, mean, median, standard deviation (sample ddof=1), min, Q1 (25%), Q3 (75%), max, range.
  - Optional extended statistical diagnostics (`extended=True`): interquartile range (IQR), Fisher-Pearson skewness, Fisher excess kurtosis, zero count, negative count, and Tukey 1.5x IQR outlier count.
  - Support for arbitrary custom percentiles (`percentiles=[0.05, 0.95]`).
- **Schema Validation (`schema_report`)**:
  - Validation against flexible schema dictionaries supporting shorthand type strings, Python/numpy types, and configuration mappings.
  - Checks for column presence, data types, nullability constraints (`nullable` / `non_null`), uniqueness constraints (`unique`), and unexpected columns (`allow_unexpected`).
  - Structured output with per-check status (`PASS` / `FAIL`) and detailed diagnostic messages.
- **Robustness & Edge-Case Handling**:
  - Complete immutability: zero in-place mutations on input DataFrames.
  - Graceful handling of empty DataFrames (`0x0`, `0xN`, `Nx0`).
  - Resilient against infinite values (`np.inf`, `-np.inf`) and unhashable cell types (lists, dicts).
  - Safe operations on DataFrames with duplicate column names and duplicate index values.
- **Comprehensive Documentation & Test Suite**:
  - 133 unit tests covering core quality functions, edge cases, enhancements, schema validation, and report container operations.
  - Standardized `src/` layout.
  - Complete documentation suite, realistic examples, and interactive documentation website.
