"""
Unit tests for missing_report() in prepix.quality.
"""
import unittest
import numpy as np
import pandas as pd
from prepix import missing_report


class TestMissingReport(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "col_0": [1, 2, 3, 4, 5],
            "col_20": [1, 2, 3, 4, None],
            "col_40": [1, 2, 3, None, None],
            "col_100": [None, None, None, None, None],
        })

    def test_missing_counts_and_percentages(self):
        res = missing_report(self.df, low_threshold=10.0, medium_threshold=30.0)
        self.assertEqual(len(res), 4)

        row_0 = res[res["Column"] == "col_0"].iloc[0]
        self.assertEqual(row_0["Missing Values"], 0)
        self.assertEqual(row_0["Missing Percentage(%)"], 0.0)
        self.assertEqual(row_0["Status"], "No Missing")
        self.assertEqual(row_0["Suggestion"], "No Action Needed")

        row_20 = res[res["Column"] == "col_20"].iloc[0]
        self.assertEqual(row_20["Missing Values"], 1)
        self.assertEqual(row_20["Missing Percentage(%)"], 20.0)
        self.assertEqual(row_20["Status"], "Medium")

        row_100 = res[res["Column"] == "col_100"].iloc[0]
        self.assertEqual(row_100["Missing Values"], 5)
        self.assertEqual(row_100["Missing Percentage(%)"], 100.0)
        self.assertEqual(row_100["Status"], "High")
        self.assertEqual(row_100["Suggestion"], "Review Before Imputation")

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            missing_report(self.df, low_threshold=-1.0)
        with self.assertRaises(ValueError):
            missing_report(self.df, low_threshold=101.0)
        with self.assertRaises(ValueError):
            missing_report(self.df, low_threshold=40.0, medium_threshold=20.0)
        with self.assertRaises(TypeError):
            missing_report(self.df, low_threshold=True)
        with self.assertRaises(TypeError):
            missing_report(self.df, low_threshold="10")

    def test_empty_dataframe(self):
        res0 = missing_report(pd.DataFrame())
        self.assertEqual(len(res0), 0)
        self.assertListEqual(list(res0.columns), ["Column", "Missing Values", "Missing Percentage(%)", "Status", "Suggestion"])

    def test_immutability(self):
        orig = self.df.copy(deep=True)
        _ = missing_report(self.df)
        pd.testing.assert_frame_equal(self.df, orig)


if __name__ == "__main__":
    unittest.main()
