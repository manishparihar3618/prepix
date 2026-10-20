"""
Unit tests for column_summary() in prepix.quality.
"""
import unittest
import numpy as np
import pandas as pd
from prepix import column_summary


class TestColumnSummary(unittest.TestCase):
    def setUp(self):
        self.df_normal = pd.DataFrame({
            "age": [20, 30, 40, 50],
            "score": [85.5, 90.0, 78.5, 92.0],
            "city": ["NY", "LA", "SF", "NY"],
            "is_active": [True, False, True, True],
        })

    def test_schema_and_columns(self):
        res = column_summary(self.df_normal)
        self.assertIsInstance(res, pd.DataFrame)
        expected_cols = [
            "Column",
            "Data Type",
            "Missing Values",
            "Missing Percentage(%)",
            "Unique Values",
            "Memory(Bytes)",
            "Suggested Type",
        ]
        self.assertListEqual(list(res.columns), expected_cols)
        self.assertEqual(len(res), 4)

    def test_suggested_types(self):
        df_types = pd.DataFrame({
            "num": [1, 2, 3],
            "bool_col": [True, False, True],
            "dt_col": pd.date_range("2025-01-01", periods=3),
            "cat_2": ["A", "B", "A"],
            "cat_multi": ["A", "B", "C"],
        })
        res = column_summary(df_types)
        type_map = dict(zip(res["Column"], res["Suggested Type"]))
        self.assertEqual(type_map["num"], "Numeric")
        self.assertEqual(type_map["bool_col"], "Binary")
        self.assertEqual(type_map["dt_col"], "Datetime")
        self.assertEqual(type_map["cat_2"], "Categorical (2 values)")
        self.assertEqual(type_map["cat_multi"], "Categorical")

    def test_empty_dataframe(self):
        res0 = column_summary(pd.DataFrame())
        self.assertEqual(len(res0), 0)
        self.assertIn("Suggested Type", res0.columns)

        res_cols = column_summary(pd.DataFrame(columns=["a", "b"]))
        self.assertEqual(len(res_cols), 2)
        self.assertTrue((res_cols["Missing Values"] == 0).all())

    def test_immutability(self):
        orig = self.df_normal.copy(deep=True)
        _ = column_summary(self.df_normal)
        pd.testing.assert_frame_equal(self.df_normal, orig)

    def test_invalid_input(self):
        for invalid in [None, 123, "df", [], {}]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    column_summary(invalid)


if __name__ == "__main__":
    unittest.main()
