"""
Performance benchmarking script for prepix quality functions.
Measures execution time across 1k, 10k, 100k rows and wide DataFrame (100+ cols).
"""
import time
import numpy as np
import pandas as pd
from prepix.quality import (
    column_summary,
    duplicate_report,
    missing_report,
    numeric_summary,
    unique_report,
)


def generate_benchmark_data(n_rows: int, n_cols: int = 10, random_seed: int = 42):
    np.random.seed(random_seed)
    data = {}
    for i in range(n_cols):
        col_type = i % 5
        if col_type == 0:
            # integer with some NaNs (float)
            arr = np.random.randint(0, 100, size=n_rows).astype(float)
            arr[np.random.rand(n_rows) < 0.1] = np.nan
            data[f"int_col_{i}"] = arr
        elif col_type == 1:
            # float with NaNs
            arr = np.random.randn(n_rows)
            arr[np.random.rand(n_rows) < 0.2] = np.nan
            data[f"float_col_{i}"] = arr
        elif col_type == 2:
            # string categories (some duplicates)
            cats = [f"cat_{k}" for k in range(5)]
            arr = np.random.choice(cats, size=n_rows).astype(object)
            arr[np.random.rand(n_rows) < 0.05] = None
            data[f"str_col_{i}"] = arr
        elif col_type == 3:
            # boolean
            data[f"bool_col_{i}"] = np.random.choice([True, False], size=n_rows)
        else:
            # datetime
            base = pd.Timestamp("2024-01-01")
            data[f"dt_col_{i}"] = [base + pd.Timedelta(days=int(x)) for x in np.random.randint(0, 1000, size=n_rows)]

    return pd.DataFrame(data)


def benchmark():
    configs = [
        ("1,000 rows (10 cols)", 1000, 10, 5),
        ("10,000 rows (10 cols)", 10000, 10, 5),
        ("100,000 rows (10 cols)", 100000, 10, 3),
        ("Wide DataFrame (100 rows, 150 cols)", 100, 150, 5),
    ]

    results = []

    for name, rows, cols, repeats in configs:
        df = generate_benchmark_data(rows, cols)

        # Warm-up
        _ = missing_report(df)
        _ = column_summary(df)
        _ = duplicate_report(df)
        _ = unique_report(df)
        _ = numeric_summary(df)

        # Benchmark missing_report
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = missing_report(df)
        t1 = time.perf_counter()
        avg_missing = (t1 - t0) / repeats * 1000.0

        # Benchmark column_summary
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = column_summary(df)
        t1 = time.perf_counter()
        avg_summary = (t1 - t0) / repeats * 1000.0

        # Benchmark duplicate_report
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = duplicate_report(df)
        t1 = time.perf_counter()
        avg_duplicate = (t1 - t0) / repeats * 1000.0

        # Benchmark unique_report
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = unique_report(df)
        t1 = time.perf_counter()
        avg_unique = (t1 - t0) / repeats * 1000.0

        # Benchmark numeric_summary
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = numeric_summary(df)
        t1 = time.perf_counter()
        avg_numeric = (t1 - t0) / repeats * 1000.0

        results.append({
            "Scenario": name,
            "Rows": rows,
            "Columns": cols,
            "missing_report (ms)": round(avg_missing, 2),
            "column_summary (ms)": round(avg_summary, 2),
            "duplicate_report (ms)": round(avg_duplicate, 2),
            "unique_report (ms)": round(avg_unique, 2),
            "numeric_summary (ms)": round(avg_numeric, 2),
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    benchmark()
