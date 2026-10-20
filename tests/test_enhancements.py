"""
Unit and regression tests for new enhancements in Prepix functions.
"""
import unittest
import numpy as np
import pandas as pd

import prepix as pp


class TestEnhancements(unittest.TestCase):
    def test_missing_report_only_missing(self):
        df = pd.DataFrame({
            "clean_1": [1, 2, 3],
            "clean_2": ["a", "b", "c"],
            "has_na": [1, None, 3],
        })
        rep_all = pp.missing_report(df, only_missing=False)
        self.assertEqual(len(rep_all), 3)

        rep_only = pp.missing_report(df, only_missing=True)
        self.assertEqual(len(rep_only), 1)
        self.assertEqual(rep_only.iloc[0]["Column"], "has_na")

    def test_missing_report_nan_threshold_rejection(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with self.assertRaises(ValueError):
            pp.missing_report(df, low_threshold=float("nan"))
        with self.assertRaises(ValueError):
            pp.missing_report(df, medium_threshold=float("nan"))

    def test_duplicate_report_subset(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 3],
            "val": ["a", "b", "c", "d"],
        })
        # Full rows have no duplicates
        rep_full = pp.duplicate_report(df)
        self.assertEqual(rep_full.iloc[0]["Duplicate Rows"], 0)

        # Subset on id has 1 duplicate
        rep_sub = pp.duplicate_report(df, subset=["id"])
        self.assertEqual(rep_sub.iloc[0]["Duplicate Rows"], 1)
        self.assertEqual(rep_sub.iloc[0]["Status"], "Duplicates Found")

        # Invalid column in subset
        with self.assertRaises(KeyError):
            pp.duplicate_report(df, subset=["non_existent"])

    def test_duplicate_report_return_records(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 3],
            "b": ["x", "y", "x", "z"],
        })
        dup_records = pp.duplicate_report(df, return_records=True)
        self.assertIsInstance(dup_records, pd.DataFrame)
        self.assertEqual(len(dup_records), 1)
        self.assertEqual(dup_records.iloc[0]["a"], 1)
        self.assertEqual(dup_records.iloc[0]["b"], "x")

    def test_duplicate_report_keep_options(self):
        # Rows 0 and 2 are identical. Row 1 is distinct.
        df = pd.DataFrame({
            "col": ["A", "B", "A"]
        })

        # keep='first' -> row 2 is duplicate (count = 1)
        rep_first = pp.duplicate_report(df, keep="first")
        self.assertEqual(rep_first.iloc[0]["Duplicate Rows"], 1)
        rec_first = pp.duplicate_report(df, keep="first", return_records=True)
        self.assertEqual(len(rec_first), 1)
        self.assertEqual(list(rec_first.index), [2])

        # keep='last' -> row 0 is duplicate (count = 1)
        rep_last = pp.duplicate_report(df, keep="last")
        self.assertEqual(rep_last.iloc[0]["Duplicate Rows"], 1)
        rec_last = pp.duplicate_report(df, keep="last", return_records=True)
        self.assertEqual(len(rec_last), 1)
        self.assertEqual(list(rec_last.index), [0])

        # keep=False -> both row 0 and row 2 are marked as duplicates (count = 2)
        rep_all = pp.duplicate_report(df, keep=False)
        self.assertEqual(rep_all.iloc[0]["Duplicate Rows"], 2)
        rec_all = pp.duplicate_report(df, keep=False, return_records=True)
        self.assertEqual(len(rec_all), 2)
        self.assertEqual(list(rec_all.index), [0, 2])

        # Invalid keep parameter
        with self.assertRaises(ValueError):
            pp.duplicate_report(df, keep="invalid")

    def test_duplicate_report_repeated_ids_with_different_data(self):
        # Repeated ID, but different names
        df = pd.DataFrame({
            "ID": [101, 101],
            "Name": ["Alice", "Bob"]
        })
        # Full row check: 0 duplicates
        rep_full = pp.duplicate_report(df)
        self.assertEqual(rep_full.iloc[0]["Duplicate Rows"], 0)

        # Subset check on ID: 1 duplicate
        rep_id = pp.duplicate_report(df, subset=["ID"])
        self.assertEqual(rep_id.iloc[0]["Duplicate Rows"], 1)
        recs = pp.duplicate_report(df, subset=["ID"], return_records=True)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs.iloc[0]["Name"], "Bob")

    def test_column_summary_dropna_consistency(self):
        df = pd.DataFrame({
            "Age": [25, 30, np.nan, 25, np.nan]
        })
        # dropna=True (default): unique ages are 25, 30 -> 2
        sum_drop = pp.column_summary(df, dropna=True)
        self.assertEqual(sum_drop.iloc[0]["Unique Values"], 2)

        # dropna=False: unique values are 25, 30, NaN -> 3
        sum_nodrop = pp.column_summary(df, dropna=False)
        self.assertEqual(sum_nodrop.iloc[0]["Unique Values"], 3)

    def test_unique_report_duplicate_count_when_dropna_true(self):
        # 5 total rows, 2 nulls, 3 valid numbers: [10, 10, 20] (2 unique, 1 duplicate)
        df = pd.DataFrame({"val": [10, 10, 20, np.nan, np.nan]})
        rep_drop = pp.unique_report(df, dropna=True)
        row = rep_drop.iloc[0]
        self.assertEqual(row["Unique Values"], 2)
        # Duplicate Values should be valid_count (3) - unique_count (2) = 1 (NOT 5 - 2 = 3)
        self.assertEqual(row["Duplicate Values"], 1)

    def test_identifier_column_not_described_as_normal_continuous(self):
        df = pd.DataFrame({
            "ID": list(range(100, 115)) + [100, 101],  # 17 rows, 15 unique
        })
        rep = pp.unique_report(df)
        suggestion = rep.iloc[0]["Suggestion"]
        # Must not claim it is a "Normal continuous numeric distribution"
        self.assertNotIn("Normal continuous numeric distribution", suggestion)
        self.assertTrue("identifier" in suggestion.lower() or "discrete" in suggestion.lower())

    def test_unified_report_unique_counts_agreement(self):
        df = pd.DataFrame({
            "Age": [28.0, 32.0, np.nan, 26.0, 45.0, np.nan, 28.0]
        })
        # Default report: dropna_unique=True ensures columns summary and unique report agree
        rep = pp.report(df)
        col_unique = rep.columns[rep.columns["Column"] == "Age"].iloc[0]["Unique Values"]
        uniq_unique = rep.unique[rep.unique["Column"] == "Age"].iloc[0]["Unique Values"]
        self.assertEqual(col_unique, uniq_unique)
        self.assertEqual(col_unique, 4)  # 28, 32, 26, 45 -> 4 non-null unique ages

    def test_numeric_summary_extended(self):
        # Dataset with known outliers: 10, 11, 12, 13, 14, 15, 100
        # Q1 = 11, Q3 = 14.5, IQR = 3.5, 100 > 14.5 + 1.5*3.5 = 19.75 (outlier!)
        # Negative values: -5
        # Zeros: 0
        vals = [0, -5, 10, 11, 12, 13, 14, 15, 100]
        df = pd.DataFrame({"num": vals})

        rep = pp.numeric_summary(df, extended=True)
        self.assertEqual(len(rep), 1)
        row = rep.iloc[0]

        # Verify extended columns exist
        for col in ["IQR", "Skewness", "Kurtosis", "Zeros", "Negatives", "Outliers"]:
            self.assertIn(col, rep.columns)

        self.assertEqual(row["Zeros"], 1)
        self.assertEqual(row["Negatives"], 1)
        self.assertGreaterEqual(row["Outliers"], 1)
        self.assertFalse(np.isnan(row["IQR"]))
        self.assertFalse(np.isnan(row["Skewness"]))
        self.assertFalse(np.isnan(row["Kurtosis"]))

    def test_numeric_summary_percentiles(self):
        df = pd.DataFrame({"num": list(range(101))})
        rep = pp.numeric_summary(df, percentiles=[0.05, 0.95])
        self.assertIn("P5", rep.columns)
        self.assertIn("P95", rep.columns)
        row = rep.iloc[0]
        self.assertAlmostEqual(row["P5"], 5.0)
        self.assertAlmostEqual(row["P95"], 95.0)

    def test_suggested_type_timedelta_and_period(self):
        df = pd.DataFrame({
            "td": pd.to_timedelta([1, 2, 3], unit="D"),
            "period": pd.period_range("2024-01", periods=3, freq="M"),
        })
        summ = pp.column_summary(df)
        types = dict(zip(summ["Column"], summ["Suggested Type"]))
        self.assertEqual(types["td"], "Timedelta")
        self.assertEqual(types["period"], "Period")


if __name__ == "__main__":
    unittest.main()
