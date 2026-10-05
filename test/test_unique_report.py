"""
Unit and regression tests for unique_report() in prepix.quality.
"""
import unittest
import numpy as np
import pandas as pd
from prepix import unique_report


class TestUniqueReport(unittest.TestCase):
    # Test 1 — Normal
    def test_normal_dataset(self):
        df = pd.DataFrame({
            "Name": ["A", "B", "C", "D", "E"],
            "Age": [20, 20, 21, 21, 20],
            "City": ["Indore", "Delhi", "Indore", "Delhi", "Indore"],
        })
        rep = unique_report(df)
        self.assertIsInstance(rep, pd.DataFrame)
        expected_cols = [
            "Column",
            "Data Type",
            "Total Values",
            "Unique Values",
            "Unique Percentage(%)",
            "Duplicate Values",
            "Cardinality",
            "Status",
            "Suggestion",
        ]
        self.assertListEqual(list(rep.columns), expected_cols)
        self.assertEqual(len(rep), 3)

        # Name: 5 rows, 5 unique
        row_name = rep[rep["Column"] == "Name"].iloc[0]
        self.assertEqual(row_name["Unique Values"], 5)
        self.assertEqual(row_name["Unique Percentage(%)"], 100.0)
        self.assertEqual(row_name["Duplicate Values"], 0)
        self.assertEqual(row_name["Cardinality"], "Very High (Unique)")
        self.assertEqual(row_name["Status"], "Potential Identifier")

        # Age: 5 rows, 2 unique
        row_age = rep[rep["Column"] == "Age"].iloc[0]
        self.assertEqual(row_age["Unique Values"], 2)
        self.assertEqual(row_age["Unique Percentage(%)"], 40.0)
        self.assertEqual(row_age["Duplicate Values"], 3)
        self.assertEqual(row_age["Cardinality"], "Medium")

        # City: 5 rows, 2 unique
        row_city = rep[rep["Column"] == "City"].iloc[0]
        self.assertEqual(row_city["Unique Values"], 2)
        self.assertEqual(row_city["Unique Percentage(%)"], 40.0)
        self.assertEqual(row_city["Duplicate Values"], 3)
        self.assertEqual(row_city["Cardinality"], "Medium")

    # Test 2 — All unique
    def test_all_unique(self):
        df = pd.DataFrame({"ID": [1, 2, 3, 4, 5]})
        rep = unique_report(df)
        row = rep.iloc[0]
        self.assertEqual(row["Unique Values"], 5)
        self.assertEqual(row["Unique Percentage(%)"], 100.0)
        self.assertEqual(row["Duplicate Values"], 0)
        self.assertEqual(row["Cardinality"], "Very High (Unique)")
        self.assertEqual(row["Status"], "Potential Identifier")

    # Test 3 — Constant
    def test_constant_column(self):
        df = pd.DataFrame({"A": [10, 10, 10, 10, 10]})
        rep = unique_report(df)
        row = rep.iloc[0]
        self.assertEqual(row["Unique Values"], 1)
        self.assertEqual(row["Unique Percentage(%)"], 20.0)
        self.assertEqual(row["Duplicate Values"], 4)
        self.assertEqual(row["Cardinality"], "Constant")
        self.assertEqual(row["Status"], "Zero Variance")

    # Test 4 — Missing values (dropna=False vs dropna=True)
    def test_missing_values(self):
        df = pd.DataFrame({"A": [1, 2, 3, None, None]})
        # dropna=False (default): 3 numbers + 1 null = 4 unique
        rep_false = unique_report(df, dropna=False)
        row_false = rep_false.iloc[0]
        self.assertEqual(row_false["Unique Values"], 4)
        self.assertEqual(row_false["Unique Percentage(%)"], 80.0)

        # dropna=True: 3 numbers = 3 unique
        rep_true = unique_report(df, dropna=True)
        row_true = rep_true.iloc[0]
        self.assertEqual(row_true["Unique Values"], 3)
        self.assertEqual(row_true["Unique Percentage(%)"], 60.0)

    # Test 5 — Empty DataFrame
    def test_empty_dataframe(self):
        # 0 rows, 0 cols
        rep0 = unique_report(pd.DataFrame())
        self.assertEqual(len(rep0), 0)
        self.assertIn("Cardinality", rep0.columns)

        # 0 rows, 2 cols
        rep_cols = unique_report(pd.DataFrame(columns=["col1", "col2"]))
        self.assertEqual(len(rep_cols), 2)
        self.assertTrue((rep_cols["Total Values"] == 0).all())
        self.assertTrue((rep_cols["Unique Values"] == 0).all())
        self.assertTrue((rep_cols["Unique Percentage(%)"] == 0.0).all())

    # Test 6 — One row
    def test_single_row(self):
        df = pd.DataFrame({"x": [100], "y": ["single"]})
        rep = unique_report(df)
        self.assertEqual(len(rep), 2)
        self.assertTrue((rep["Total Values"] == 1).all())
        self.assertTrue((rep["Unique Values"] == 1).all())

    # Test 7 — Mixed dtypes
    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4],
            "float_col": [1.1, 2.2, 3.3, 4.4],
            "str_col": ["a", "b", "c", "d"],
            "bool_col": [True, False, True, False],
            "dt_col": pd.date_range("2025-01-01", periods=4),
            "cat_col": pd.Categorical(["cat", "dog", "cat", "bird"]),
        })
        rep = unique_report(df)
        self.assertEqual(len(rep), 6)
        self.assertListEqual(list(rep["Column"]), ["int_col", "float_col", "str_col", "bool_col", "dt_col", "cat_col"])

    # Test 8 — 100+ columns
    def test_large_column_count(self):
        data = {f"col_{i}": np.random.randn(20) for i in range(120)}
        df_wide = pd.DataFrame(data)
        rep = unique_report(df_wide)
        self.assertEqual(len(rep), 120)

    # Test 9 — High-cardinality text
    def test_high_cardinality_text(self):
        df_text = pd.DataFrame({"text_id": [f"user_{i}" for i in range(100)]})
        rep = unique_report(df_text)
        row = rep.iloc[0]
        self.assertEqual(row["Cardinality"], "Very High (Unique)")
        self.assertEqual(row["Status"], "Potential Identifier")

    # Test 10 — Invalid inputs
    def test_invalid_inputs(self):
        for invalid in [None, [1, 2], "data", 123, {"a": 1}, (1, 2), pd.Series([1, 2])]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    unique_report(invalid)

        with self.assertRaises(TypeError):
            unique_report(pd.DataFrame({"a": [1, 2]}), dropna="not_a_bool")

    # Test 11 — Original DataFrame unchanged
    def test_dataframe_immutability(self):
        df = pd.DataFrame({"a": [1, 2, 3, 2, 1], "b": ["x", "y", "z", "y", "x"]})
        orig = df.copy(deep=True)
        _ = unique_report(df)
        pd.testing.assert_frame_equal(df, orig)


if __name__ == "__main__":
    unittest.main()
