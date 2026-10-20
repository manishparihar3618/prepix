"""
Edge case and stress tests for prepix quality functions.
"""
import unittest
import numpy as np
import pandas as pd
from prepix import column_summary, duplicate_report, missing_report


class TestEdgeCases(unittest.TestCase):
    def test_single_row_single_column(self):
        df_1x1 = pd.DataFrame({"only": [999]})
        self.assertEqual(len(column_summary(df_1x1)), 1)
        self.assertEqual(len(missing_report(df_1x1)), 1)
        self.assertEqual(len(duplicate_report(df_1x1)), 1)

    def test_all_missing_dataframe(self):
        df_nan = pd.DataFrame({
            "a": [np.nan, np.nan],
            "b": [None, None],
        })
        rep = missing_report(df_nan)
        self.assertTrue((rep["Missing Percentage(%)"] == 100.0).all())
        self.assertTrue((rep["Status"] == "High").all())

    def test_special_numerical_values(self):
        df_special = pd.DataFrame({
            "inf": [np.inf, -np.inf, 0.0],
            "extreme": [1e308, -1e308, 1e-308],
        })
        rep_miss = missing_report(df_special)
        rep_sum = column_summary(df_special)
        self.assertEqual(len(rep_miss), 2)
        self.assertEqual(len(rep_sum), 2)

    def test_unusual_column_headers(self):
        df_headers = pd.DataFrame({
            "col with space": [1],
            "123": [2],
            "unicode_🚀_ñ": [3],
            "!@#$%^&*()": [4],
        })
        self.assertEqual(len(missing_report(df_headers)), 4)
        self.assertEqual(len(column_summary(df_headers)), 4)
        self.assertEqual(len(duplicate_report(df_headers)), 1)

    def test_duplicate_column_headers(self):
        df_dups = pd.DataFrame([[1, "a"], [None, "b"]], columns=["dup", "dup"])
        self.assertEqual(len(missing_report(df_dups)), 2)
        self.assertEqual(len(column_summary(df_dups)), 2)
        self.assertEqual(len(duplicate_report(df_dups)), 1)


if __name__ == "__main__":
    unittest.main()
