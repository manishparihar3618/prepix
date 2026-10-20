"""
Prepix Example: Multi-Source DataFrame Ingestion and Analysis
Demonstrates that Prepix operates seamlessly on pandas DataFrames regardless of
their original data source (in-memory, CSV, JSON, SQLite, Parquet, Excel).

Note: Prepix analyzes DataFrames in memory; pandas handles file/database I/O.
"""
import io
import json
import os
import sqlite3
import tempfile
import pandas as pd
import prepix as pp


def demo_in_memory():
    print("\n--- 1. In-Memory DataFrame (pd.DataFrame) ---")
    df = pd.DataFrame({
        "employee_id": [101, 102, 103, 104],
        "name": ["Aman", "Priya", "Rahul", "Sneha"],
        "rating": [4.5, 3.8, None, 4.9],
        "active": [True, True, False, True],
    })
    print(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    summary = pp.column_summary(df)
    print(summary[["Column", "Data Type", "Missing Values", "Unique Values", "Suggested Type"]])
    return df


def demo_csv():
    print("\n--- 2. CSV Ingestion (pd.read_csv) ---")
    csv_data = """id,department,salary,status
1,Engineering,75000,Active
2,Marketing,62000,Active
3,HR,,Inactive
4,Engineering,75000,Active
"""
    df = pd.read_csv(io.StringIO(csv_data))
    print(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    miss = pp.missing_report(df)
    print("Missing report:\n", miss[["Column", "Missing Values", "Missing Percentage(%)", "Status"]])
    dups = pp.duplicate_report(df)
    print("Duplicate report:\n", dups[["Total Rows", "Duplicate Rows", "Unique Rows", "Status"]])
    return df


def demo_json():
    print("\n--- 3. JSON Ingestion (pd.read_json) ---")
    json_data = json.dumps([
        {"tx_id": "T001", "amount": 150.50, "currency": "USD"},
        {"tx_id": "T002", "amount": 99.00, "currency": "USD"},
        {"tx_id": "T003", "amount": 0.00, "currency": "EUR"},
        {"tx_id": "T004", "amount": -15.00, "currency": "USD"},
    ])
    df = pd.read_json(io.StringIO(json_data))
    print(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    num_summary = pp.numeric_summary(df, extended=True)
    print("Numeric summary:\n", num_summary[["Column", "Mean", "Median", "Min", "Max", "Zeros", "Negatives"]])
    return df


def demo_sqlite():
    print("\n--- 4. SQLite Database Query (pd.read_sql) ---")
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE products (product_id INT, name TEXT, price REAL, stock INT)")
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [
            (1, "Widget A", 19.99, 100),
            (2, "Widget B", 29.99, 0),
            (3, "Widget C", None, 50),
        ],
    )
    conn.commit()

    df = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    print(f"Loaded DataFrame from SQLite: {df.shape[0]} rows x {df.shape[1]} columns")

    schema = {
        "product_id": {"type": "integer", "nullable": False, "unique": True},
        "name": "string",
        "price": {"type": "float", "nullable": True},
        "stock": {"type": "integer", "nullable": False},
    }
    schema_res = pp.schema_report(df, schema=schema)
    print("Schema contract audit:\n", schema_res[["Column", "Check", "Expected", "Actual", "Status"]])
    return df


def demo_parquet():
    print("\n--- 5. Parquet Ingestion (pd.read_parquet) ---")
    try:
        import pyarrow  # noqa: F401
        has_engine = True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            has_engine = True
        except ImportError:
            has_engine = False

    if not has_engine:
        print("STATUS: SKIPPED (Optional Parquet engine 'pyarrow' or 'fastparquet' is not installed).")
        print("Guidance: Install 'pyarrow>=10.0.0' or 'fastparquet' to read parquet files into pandas.")
        print("          Prepix requires no extra configuration; once loaded as a DataFrame, all Prepix functions work identically.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_seed = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.5, 6.0]})
        df_seed.to_parquet(tmp_path)
        df = pd.read_parquet(tmp_path)
        print(f"Loaded DataFrame from Parquet: {df.shape[0]} rows x {df.shape[1]} columns")
        rep = pp.report(df)
        print("Unified report generated successfully.")
        return df
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def demo_excel():
    print("\n--- 6. Excel Workbook Ingestion (pd.read_excel) ---")
    try:
        import openpyxl  # noqa: F401
        has_engine = True
    except ImportError:
        has_engine = False

    if not has_engine:
        print("STATUS: SKIPPED (Optional Excel engine 'openpyxl' is not installed).")
        print("Guidance: Install 'openpyxl>=3.0.0' or 'calamine' to read .xlsx workbooks into pandas.")
        print("          Prepix requires no extra configuration; once loaded as a DataFrame, all Prepix functions work identically.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_seed = pd.DataFrame({"sku": ["SKU1", "SKU2"], "qty": [10, 20]})
        df_seed.to_excel(tmp_path, index=False)
        df = pd.read_excel(tmp_path)
        print(f"Loaded DataFrame from Excel: {df.shape[0]} rows x {df.shape[1]} columns")
        rep = pp.report(df)
        print("Unified report generated successfully.")
        return df
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    print("=" * 70)
    print(" PREPIX DATA SOURCES DEMONSTRATION")
    print("=" * 70)
    print("Key Architecture Principle:")
    print("Prepix is an in-memory DataFrame analysis engine.")
    print("It operates on standard pandas DataFrames regardless of how pandas ingested the data.")

    demo_in_memory()
    demo_csv()
    demo_json()
    demo_sqlite()
    demo_parquet()
    demo_excel()

    print("\n" + "=" * 70)
    print("Data sources demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
