"""
Comprehensive unit and integration tests for numeric_summary() in prepix.quality.
Covers:
- Standard statistics & independent verification (Mean, Median, Std, Min, Q1, Q3, Max, Range)
- Missing values (0%, 10%, 50%, 90%, 100% missing)
- Constant columns
- Negative values and zeros
- High decimal precision & large values (scientific notation / large int)
- Infinities (np.inf, -np.inf)
- Dtype filtering (strings, datetimes, booleans, categoricals, timedeltas excluded)
- Empty & no-numeric DataFrames
- Single row dataset (Std = NaN check)
- Mixed numerical types & nullable integers/floats
- Wide (100, 500 cols) and Large (100k+ rows) scaling
- Input validation (TypeError for non-DataFrame inputs)
- Original DataFrame immutability guarantee
"""
import unittest
import numpy as np
import pandas as pd

from prepix import (
    column_summary,
    duplicate_report,
    missing_report,
    numeric_summary,
    unique_report,
)


class TestNumericSummary(unittest.TestCase):
    # =========================================================================
    # 1. SCHEMA AND STATISTICAL CORRECTNESS (Independent Verification)
    # =========================================================================

    def test_schema_and_column_names(self):
        df = pd.DataFrame({
            "age": [20, 30, 40, 50],
            "score": [85.5, 90.0, 78.5, 92.0],
            "name": ["Alice", "Bob", "Charlie", "David"],
        })
        res = numeric_summary(df)
        self.assertIsInstance(res, pd.DataFrame)
        expected_cols = [
            "Column",
            "Data Type",
            "Count",
            "Missing Values",
            "Mean",
            "Median",
            "Std",
            "Min",
            "Q1",
            "Q3",
            "Max",
            "Range",
        ]
        self.assertListEqual(list(res.columns), expected_cols)
        self.assertEqual(len(res), 2)
        self.assertListEqual(list(res["Column"]), ["age", "score"])

    def test_independent_statistical_verification(self):
        np.random.seed(42)
        values = [12.5, 45.2, 78.9, 23.1, 89.4, 56.7, 34.2, 90.1, 11.0, 67.8]
        df = pd.DataFrame({"metric": values})

        res = numeric_summary(df)
        row = res.iloc[0]

        s = df["metric"]
        self.assertEqual(row["Column"], "metric")
        self.assertEqual(row["Count"], len(values))
        self.assertEqual(row["Missing Values"], 0)
        self.assertAlmostEqual(row["Mean"], s.mean(), places=6)
        self.assertAlmostEqual(row["Median"], s.median(), places=6)
        self.assertAlmostEqual(row["Std"], s.std(ddof=1), places=6)
        self.assertAlmostEqual(row["Min"], s.min(), places=6)
        self.assertAlmostEqual(row["Q1"], s.quantile(0.25), places=6)
        self.assertAlmostEqual(row["Q3"], s.quantile(0.75), places=6)
        self.assertAlmostEqual(row["Max"], s.max(), places=6)
        self.assertAlmostEqual(row["Range"], s.max() - s.min(), places=6)

    # =========================================================================
    # 2. MISSING VALUES GRADIENT (0%, 10%, 50%, 90%, 100%)
    # =========================================================================

    def test_missing_values_gradient(self):
        df = pd.DataFrame({
            "col_0pct": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "col_10pct": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, np.nan],
            "col_50pct": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "col_90pct": [10.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "col_100pct": [np.nan] * 10,
        })
        res = numeric_summary(df)
        self.assertEqual(len(res), 5)

        r0 = res[res["Column"] == "col_0pct"].iloc[0]
        self.assertEqual(r0["Count"], 10)
        self.assertEqual(r0["Missing Values"], 0)
        self.assertAlmostEqual(r0["Mean"], 55.0)

        r10 = res[res["Column"] == "col_10pct"].iloc[0]
        self.assertEqual(r10["Count"], 9)
        self.assertEqual(r10["Missing Values"], 1)
        self.assertAlmostEqual(r10["Mean"], 50.0)

        r50 = res[res["Column"] == "col_50pct"].iloc[0]
        self.assertEqual(r50["Count"], 5)
        self.assertEqual(r50["Missing Values"], 5)
        self.assertAlmostEqual(r50["Mean"], 30.0)

        r90 = res[res["Column"] == "col_90pct"].iloc[0]
        self.assertEqual(r90["Count"], 1)
        self.assertEqual(r90["Missing Values"], 9)
        self.assertAlmostEqual(r90["Mean"], 10.0)
        self.assertTrue(np.isnan(r90["Std"]))  # 1 observation -> std is NaN

        r100 = res[res["Column"] == "col_100pct"].iloc[0]
        self.assertEqual(r100["Count"], 0)
        self.assertEqual(r100["Missing Values"], 10)
        self.assertTrue(np.isnan(r100["Mean"]))
        self.assertTrue(np.isnan(r100["Median"]))
        self.assertTrue(np.isnan(r100["Std"]))
        self.assertTrue(np.isnan(r100["Min"]))
        self.assertTrue(np.isnan(r100["Q1"]))
        self.assertTrue(np.isnan(r100["Q3"]))
        self.assertTrue(np.isnan(r100["Max"]))
        self.assertTrue(np.isnan(r100["Range"]))

    def test_all_missing_column_does_not_crash(self):
        # 100% missing with default Python None (inferred as object dtype by pandas)
        df_none = pd.DataFrame({"Age": [None, None, None, None]})
        res_none = numeric_summary(df_none)
        self.assertIsInstance(res_none, pd.DataFrame)
        self.assertEqual(len(res_none), 0)

        # 100% missing with numpy NaN / float dtype
        df_nan = pd.DataFrame({"Age": [np.nan, np.nan, np.nan, np.nan]})
        res_nan = numeric_summary(df_nan)
        self.assertEqual(len(res_nan), 1)
        row = res_nan.iloc[0]
        self.assertEqual(row["Count"], 0)
        self.assertEqual(row["Missing Values"], 4)
        self.assertTrue(np.isnan(row["Mean"]))
        self.assertTrue(np.isnan(row["Range"]))

    # =========================================================================
    # 3. SPECIAL NUMERICAL PATTERNS (Constant, Negative, Zeros, Decimals, Large, Inf)
    # =========================================================================

    def test_constant_column(self):
        df = pd.DataFrame({"Age": [20, 20, 20, 20, 20]})
        res = numeric_summary(df)
        row = res.iloc[0]
        self.assertEqual(row["Count"], 5)
        self.assertEqual(row["Missing Values"], 0)
        self.assertAlmostEqual(row["Mean"], 20.0)
        self.assertAlmostEqual(row["Median"], 20.0)
        self.assertAlmostEqual(row["Std"], 0.0)
        self.assertAlmostEqual(row["Min"], 20.0)
        self.assertAlmostEqual(row["Q1"], 20.0)
        self.assertAlmostEqual(row["Q3"], 20.0)
        self.assertAlmostEqual(row["Max"], 20.0)
        self.assertAlmostEqual(row["Range"], 0.0)

    def test_negative_values(self):
        df = pd.DataFrame({"Temperature": [-10, -5, 0, 5, 10]})
        res = numeric_summary(df)
        row = res.iloc[0]
        self.assertEqual(row["Count"], 5)
        self.assertAlmostEqual(row["Mean"], 0.0)
        self.assertAlmostEqual(row["Median"], 0.0)
        self.assertAlmostEqual(row["Min"], -10.0)
        self.assertAlmostEqual(row["Max"], 10.0)
        self.assertAlmostEqual(row["Range"], 20.0)

    def test_zero_values(self):
        df = pd.DataFrame({"Zeros": [0, 0, 0, 0, 10]})
        res = numeric_summary(df)
        row = res.iloc[0]
        self.assertEqual(row["Count"], 5)
        self.assertEqual(row["Missing Values"], 0)
        self.assertAlmostEqual(row["Min"], 0.0)
        self.assertAlmostEqual(row["Median"], 0.0)
        self.assertAlmostEqual(row["Mean"], 2.0)

    def test_decimal_precision(self):
        values = [1.2, 2.5, 3.75, 4.1, 5.99]
        df = pd.DataFrame({"Score": values})
        res = numeric_summary(df)
        row = res.iloc[0]
        s = df["Score"]
        self.assertAlmostEqual(row["Mean"], s.mean(), places=8)
        self.assertAlmostEqual(row["Median"], s.median(), places=8)
        self.assertAlmostEqual(row["Std"], s.std(), places=8)
        self.assertAlmostEqual(row["Min"], 1.2, places=8)
        self.assertAlmostEqual(row["Max"], 5.99, places=8)
        self.assertAlmostEqual(row["Range"], 5.99 - 1.2, places=8)

    def test_large_integers_and_scientific_floats(self):
        df = pd.DataFrame({
            "large_int": [10**14, 2 * 10**14, 3 * 10**14],
            "sci_float": [1e15, 2.5e15, 5e15],
            "tiny_float": [1e-12, 2e-12, 3e-12],
        })
        res = numeric_summary(df)
        self.assertEqual(len(res), 3)
        row_int = res[res["Column"] == "large_int"].iloc[0]
        self.assertAlmostEqual(row_int["Mean"], 2 * 10**14)
        self.assertAlmostEqual(row_int["Range"], 2 * 10**14)

    def test_infinity_values(self):
        df = pd.DataFrame({"A": [1.0, 2.0, np.inf, 4.0, -np.inf]})
        res = numeric_summary(df)
        row = res.iloc[0]
        self.assertEqual(row["Count"], 5)
        self.assertEqual(row["Missing Values"], 0)
        self.assertEqual(row["Min"], -np.inf)
        self.assertEqual(row["Max"], np.inf)
        self.assertEqual(row["Range"], np.inf)

    # =========================================================================
    # 4. COLUMN TYPE FILTERING (Only Numerical Included)
    # =========================================================================

    def test_non_numerical_columns_excluded(self):
        df = pd.DataFrame({
            "Age": [20, 21, 22],
            "Name": ["A", "B", "C"],
            "City": ["Indore", "Delhi", "Bhopal"],
            "Active": [True, False, True],
            "NullableBool": pd.Series([True, False, None], dtype="boolean"),
            "Date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "Category": pd.Categorical(["cat", "dog", "cat"]),
            "Timedelta": pd.to_timedelta([1, 2, 3], unit="D"),
        })
        res = numeric_summary(df)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["Column"], "Age")

    def test_no_numerical_columns(self):
        df = pd.DataFrame({
            "Name": ["A", "B", "C"],
            "City": ["Indore", "Delhi", "Bhopal"],
            "Flag": [True, False, True],
        })
        res = numeric_summary(df)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(res), 0)
        expected_cols = [
            "Column", "Data Type", "Count", "Missing Values", "Mean", "Median",
            "Std", "Min", "Q1", "Q3", "Max", "Range"
        ]
        self.assertListEqual(list(res.columns), expected_cols)

    def test_empty_dataframes(self):
        # 0x0
        res0 = numeric_summary(pd.DataFrame())
        self.assertEqual(len(res0), 0)
        self.assertIn("Range", res0.columns)

        # 0x2 non-numerical
        res_non_num = numeric_summary(pd.DataFrame(columns=["A", "B"]))
        self.assertEqual(len(res_non_num), 0)

        # 0x2 numerical
        df_empty_num = pd.DataFrame({
            "num1": pd.Series([], dtype="float64"),
            "num2": pd.Series([], dtype="int64"),
        })
        res_num = numeric_summary(df_empty_num)
        self.assertEqual(len(res_num), 2)
        self.assertTrue((res_num["Count"] == 0).all())
        self.assertTrue((res_num["Missing Values"] == 0).all())
        self.assertTrue(np.isnan(res_num["Mean"]).all())

    # =========================================================================
    # 5. SINGLE ROW AND BOUNDARY DATASETS
    # =========================================================================

    def test_single_row(self):
        df = pd.DataFrame({"Age": [20]})
        res = numeric_summary(df)
        row = res.iloc[0]
        self.assertEqual(row["Count"], 1)
        self.assertEqual(row["Missing Values"], 0)
        self.assertAlmostEqual(row["Mean"], 20.0)
        self.assertAlmostEqual(row["Median"], 20.0)
        self.assertTrue(np.isnan(row["Std"]))  # Pandas standard ddof=1 produces NaN
        self.assertAlmostEqual(row["Min"], 20.0)
        self.assertAlmostEqual(row["Q1"], 20.0)
        self.assertAlmostEqual(row["Q3"], 20.0)
        self.assertAlmostEqual(row["Max"], 20.0)
        self.assertAlmostEqual(row["Range"], 0.0)

    def test_mixed_numerical_dtypes(self):
        df = pd.DataFrame({
            "i8": np.array([1, 2, 3], dtype=np.int8),
            "i16": np.array([10, 20, 30], dtype=np.int16),
            "i32": np.array([100, 200, 300], dtype=np.int32),
            "i64": np.array([1000, 2000, 3000], dtype=np.int64),
            "u8": np.array([1, 2, 3], dtype=np.uint8),
            "f32": np.array([1.5, 2.5, 3.5], dtype=np.float32),
            "f64": np.array([1.1, 2.2, 3.3], dtype=np.float64),
            "null_int": pd.Series([10, None, 30], dtype="Int64"),
            "null_float": pd.Series([1.5, None, 3.5], dtype="Float64"),
        })
        res = numeric_summary(df)
        self.assertEqual(len(res), 9)
        self.assertEqual(res[res["Column"] == "null_int"].iloc[0]["Missing Values"], 1)
        self.assertEqual(res[res["Column"] == "null_int"].iloc[0]["Count"], 2)

    # =========================================================================
    # 6. STRUCTURAL AND POSITION-BASED EDGE CASES
    # =========================================================================

    def test_duplicate_column_headers(self):
        df = pd.DataFrame([[10, 20.0], [30, 40.0]], columns=["val", "val"])
        res = numeric_summary(df)
        self.assertEqual(len(res), 2)
        self.assertEqual(res.iloc[0]["Mean"], 20.0)
        self.assertEqual(res.iloc[1]["Mean"], 30.0)

    def test_integer_and_tuple_column_headers(self):
        df_int = pd.DataFrame([[1, 2], [3, 4]], columns=[101, 202])
        res_int = numeric_summary(df_int)
        self.assertEqual(len(res_int), 2)
        self.assertListEqual(list(res_int["Column"]), [101, 202])

        df_multi = pd.DataFrame([[1, 2], [3, 4]], columns=pd.MultiIndex.from_tuples([("A", "x"), ("B", "y")]))
        res_multi = numeric_summary(df_multi)
        self.assertEqual(len(res_multi), 2)

    # =========================================================================
    # 7. SCALE AND STRESS TESTS (Wide & Large)
    # =========================================================================

    def test_wide_dataframe(self):
        n_cols = 100
        data = {f"num_{i}": np.random.randn(20) for i in range(n_cols)}
        data["str_col"] = ["text"] * 20
        df = pd.DataFrame(data)

        res = numeric_summary(df)
        self.assertEqual(len(res), n_cols)
        self.assertEqual(len(set(res["Column"])), n_cols)
        self.assertEqual(res.iloc[0]["Column"], "num_0")
        self.assertEqual(res.iloc[-1]["Column"], f"num_{n_cols-1}")

    def test_large_dataframe(self):
        n_rows = 100_000
        df = pd.DataFrame({
            "metric_1": np.arange(n_rows, dtype=np.float64),
            "metric_2": np.random.randn(n_rows),
            "category": np.random.choice(["A", "B", "C"], size=n_rows),
        })
        df.loc[::10, "metric_1"] = np.nan

        res = numeric_summary(df)
        self.assertEqual(len(res), 2)
        row1 = res[res["Column"] == "metric_1"].iloc[0]
        self.assertEqual(row1["Missing Values"], 10_000)
        self.assertEqual(row1["Count"], 90_000)

    # =========================================================================
    # 8. INPUT VALIDATION & IMMUTABILITY
    # =========================================================================

    def test_input_validation(self):
        for invalid in [None, [1, 2, 3], "hello", 123, pd.Series([1, 2, 3]), {"a": [1, 2]}]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    numeric_summary(invalid)

    def test_dataframe_immutability(self):
        df = pd.DataFrame({
            "A": [10, 20, np.nan, 40],
            "B": [1.5, 2.5, 3.5, 4.5],
            "C": ["x", "y", "z", "w"],
        })
        orig = df.copy(deep=True)
        _ = numeric_summary(df)
        pd.testing.assert_frame_equal(df, orig)


if __name__ == "__main__":
    unittest.main()
