"""
Unit tests for duplicate_report() in prepix.quality.
"""
import unittest
import numpy as np
import pandas as pd
from prepix import duplicate_report


class TestDuplicateReport(unittest.TestCase):
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        res = duplicate_report(df)
        self.assertEqual(len(res), 1)
        row = res.iloc[0]
        self.assertEqual(row["Total Rows"], 3)
        self.assertEqual(row["Duplicate Rows"], 0)
        self.assertEqual(row["Duplicate Percentage(%)"], 0.0)
        self.assertEqual(row["Unique Rows"], 3)
        self.assertEqual(row["Status"], "No Duplicates")
        self.assertEqual(row["Suggestion"], "No action required")

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 1, 2], "b": ["x", "y", "x", "y"]})
        res = duplicate_report(df)
        row = res.iloc[0]
        self.assertEqual(row["Total Rows"], 4)
        self.assertEqual(row["Duplicate Rows"], 2)
        self.assertEqual(row["Duplicate Percentage(%)"], 50.0)
        self.assertEqual(row["Unique Rows"], 2)
        self.assertEqual(row["Status"], "Duplicates Found")
        self.assertEqual(row["Suggestion"], "Consider reviewing and removing duplicate records")

    def test_empty_dataframe(self):
        res = duplicate_report(pd.DataFrame())
        row = res.iloc[0]
        self.assertEqual(row["Total Rows"], 0)
        self.assertEqual(row["Duplicate Rows"], 0)
        self.assertEqual(row["Duplicate Percentage(%)"], 0.0)
        self.assertEqual(row["Unique Rows"], 0)
        self.assertEqual(row["Status"], "No Duplicates")

    def test_immutability(self):
        df = pd.DataFrame({"x": [1, 2, 1]})
        orig = df.copy(deep=True)
        _ = duplicate_report(df)
        pd.testing.assert_frame_equal(df, orig)


if __name__ == "__main__":
    unittest.main()
