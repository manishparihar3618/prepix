# Data Sources Guide

This guide explains how **Prepix** integrates with various data sources across the Python and pandas ecosystem.

---

## Architecture Principle: Prepix Analyzes DataFrames

**Prepix is an in-memory data quality, profiling, and validation engine.** It operates exclusively on pandas `DataFrame` objects.

Prepix intentionally does **not** reimplement file parsing or database communication. Instead, you use the standard, battle-tested pandas I/O functions (`pd.read_csv`, `pd.read_sql`, etc.) to ingest your data, and then pass the resulting DataFrame directly to Prepix functions:

```text
[ Data Source ]  -->  [ pandas I/O (read_*) ]  -->  [ pd.DataFrame ]  -->  [ Prepix Audit ]
  - CSV Files           pd.read_csv()                  In-Memory              pp.report(df)
  - JSON APIs           pd.read_json()                                        pp.missing_report(df)
  - SQL Databases       pd.read_sql()                                         pp.schema_report(df, schema)
  - Parquet Files       pd.read_parquet()
  - Excel Sheets        pd.read_excel()
```

---

## 1. In-Memory DataFrames

Construct DataFrames directly from Python dictionaries, lists, or numpy arrays:

```python
import pandas as pd
import prepix as pp

df = pd.DataFrame({
    "user_id": [101, 102, 103, 104],
    "name": ["Aman", "Priya", "Rahul", "Sneha"],
    "rating": [4.8, 3.9, None, 4.5],
    "is_active": [True, True, False, True]
})

print(pp.column_summary(df))
print(pp.report(df))
```

*Required Dependencies*: `pandas>=2.0.0`, `numpy>=1.24.0` (core dependencies).

---

## 2. CSV Files (`pd.read_csv`)

CSV is the most common format for tabular data. Pandas includes a built-in C-optimized CSV parser:

```python
import pandas as pd
import prepix as pp

# Ingest CSV into DataFrame
df = pd.read_csv("data/customer_transactions.csv")

# Audit data quality with Prepix
missing = pp.missing_report(df, only_missing=True)
duplicates = pp.duplicate_report(df)

print(missing)
print(duplicates)
```

*Required Dependencies*: Built into standard pandas. No external dependencies required.

---

## 3. JSON Files and API Responses (`pd.read_json`)

Ingest records from REST APIs, event streams, or document dumps:

```python
import pandas as pd
import prepix as pp

# Read JSON file or response string
df = pd.read_json("data/events.json")

# Validate numerical statistics and distributions
num_stats = pp.numeric_summary(df, extended=True)
print(num_stats)
```

*Required Dependencies*: Built into standard pandas.

---

## 4. SQL Databases (`pd.read_sql`)

Query relational databases (SQLite, PostgreSQL, MySQL, Snowflake) using Python database connectors or SQLAlchemy:

### SQLite Example (Zero extra dependencies)
```python
import sqlite3
import pandas as pd
import prepix as pp

# Connect to database
conn = sqlite3.connect("production.db")

# Ingest query results into DataFrame
df = pd.read_sql("SELECT order_id, customer_name, total_amount, order_status FROM orders", conn)
conn.close()

# Verify expected data contract
schema = {
    "order_id": {"type": "integer", "nullable": False, "unique": True},
    "customer_name": "string",
    "total_amount": {"type": "numeric", "nullable": False},
    "order_status": "string",
}

validation = pp.schema_report(df, schema=schema)
print(validation)
```

*Required Dependencies*: `sqlite3` is built into the Python standard library. Other engines (PostgreSQL, MySQL) require their respective drivers (e.g., `psycopg2`, `pymysql`, `SQLAlchemy`).

---

## 5. Parquet Files (`pd.read_parquet`)

Parquet is a columnar storage format widely used in Apache Spark, Databricks, and cloud data lakes (AWS S3, GCP Cloud Storage):

```python
import pandas as pd
import prepix as pp

# Ingest Parquet dataset
df = pd.read_parquet("data/telemetry.parquet")

# Run comprehensive dataset quality audit
report = pp.report(df)
print(report)
```

*Required Dependencies*:
- `pyarrow>=10.0.0` (recommended) OR `fastparquet`
- Install via: `pip install pyarrow`

---

## 6. Excel Workbooks (`pd.read_excel`)

Ingest spreadsheets (`.xlsx`, `.xls`):

```python
import pandas as pd
import prepix as pp

# Ingest Excel sheet
df = pd.read_excel("data/quarterly_budget.xlsx", sheet_name="Q4_Actuals")

# Inspect column dtypes, uniqueness, and memory
print(pp.column_summary(df))
```

*Required Dependencies*:
- `openpyxl>=3.0.0` (for `.xlsx`) OR `calamine`
- Install via: `pip install openpyxl`

---

## Summary of Data Source Requirements

| Source Format | Pandas Ingestion Function | Required Extra Dependency | Prepix Configuration Needed? |
| :--- | :--- | :--- | :--- |
| **In-Memory** | `pd.DataFrame(...)` | None | None |
| **CSV** | `pd.read_csv(...)` | None | None |
| **JSON** | `pd.read_json(...)` | None | None |
| **SQLite** | `pd.read_sql(...)` | None (`sqlite3` in standard lib) | None |
| **Postgres / MySQL** | `pd.read_sql(...)` | `psycopg2`, `pymysql`, `SQLAlchemy` | None |
| **Parquet** | `pd.read_parquet(...)` | `pyarrow` or `fastparquet` | None |
| **Excel** | `pd.read_excel(...)` | `openpyxl` or `calamine` | None |

Once loaded into pandas as a DataFrame, every Prepix function (`report`, `missing_report`, `column_summary`, `duplicate_report`, `unique_report`, `numeric_summary`, `schema_report`) works identically and guarantees complete DataFrame immutability.
