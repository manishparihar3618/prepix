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

    def test_keep_semantics_and_counts(self):
        # 3 identical rows
        df = pd.DataFrame({"val": [1, 1, 1]})

        # keep="first": 2 duplicate copies, 1 distinct remaining entity
        res_first = duplicate_report(df, keep="first")
        self.assertEqual(res_first.iloc[0]["Duplicate Rows"], 2)
        self.assertEqual(res_first.iloc[0]["Unique Rows"], 1)
        recs_first = duplicate_report(df, keep="first", return_records=True)
        self.assertEqual(len(recs_first), 2)

        # keep="last": 2 duplicate copies, 1 distinct remaining entity
        res_last = duplicate_report(df, keep="last")
        self.assertEqual(res_last.iloc[0]["Duplicate Rows"], 2)
        self.assertEqual(res_last.iloc[0]["Unique Rows"], 1)
        recs_last = duplicate_report(df, keep="last", return_records=True)
        self.assertEqual(len(recs_last), 2)

        # keep=False: all 3 rows participate in duplicate group
        res_false = duplicate_report(df, keep=False)
        self.assertEqual(res_false.iloc[0]["Duplicate Rows"], 3)
        self.assertEqual(res_false.iloc[0]["Unique Rows"], 0)
        recs_false = duplicate_report(df, keep=False, return_records=True)
        self.assertEqual(len(recs_false), 3)

    def test_unhashable_cell_values(self):
        df = pd.DataFrame({
            "lists": [[1, 2], [3, 4], [1, 2]],
            "dicts": [{"a": 1}, {"b": 2}, {"a": 1}],
        })
        res = duplicate_report(df)
        self.assertEqual(res.iloc[0]["Duplicate Rows"], 1)
        self.assertEqual(res.iloc[0]["Unique Rows"], 2)

        recs = duplicate_report(df, return_records=True)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs.iloc[0]["lists"], [1, 2])
        self.assertEqual(recs.iloc[0]["dicts"], {"a": 1})

    def test_non_unique_index_labels(self):
        df = pd.DataFrame(
            {"col": ["A", "B", "A"]},
            index=["dup_lbl", "unique_lbl", "dup_lbl"]
        )
        res = duplicate_report(df)
        self.assertEqual(res.iloc[0]["Duplicate Rows"], 1)

        recs = duplicate_report(df, return_records=True)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs.index.tolist(), ["dup_lbl"])
        self.assertEqual(recs.iloc[0]["col"], "A")

    def test_duplicate_column_headers(self):
        df = pd.DataFrame(
            [[10, 20], [10, 20], [30, 40]],
            columns=["same", "same"]
        )
        res = duplicate_report(df)
        self.assertEqual(res.iloc[0]["Total Rows"], 3)
        self.assertEqual(res.iloc[0]["Duplicate Rows"], 1)
        self.assertEqual(res.iloc[0]["Unique Rows"], 2)

        recs = duplicate_report(df, return_records=True)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs.iloc[0].to_list(), [10, 20])


if __name__ == "__main__":
    unittest.main()

