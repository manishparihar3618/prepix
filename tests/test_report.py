"""
Unit and integration tests for prepix.report() and PrepixReport class.
"""
import json
import unittest
import numpy as np
import pandas as pd

import prepix as pp
from prepix import PrepixReport, report


class TestPrepixReport(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "age": [25, 30, 35, 40, np.nan, 200],  # 1 missing, 1 outlier (200)
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0],
            "dept": ["HR", "IT", "IT", "HR", "Finance", "HR"],
            "flag": [True, True, True, True, True, True],  # Constant boolean
        })

    def test_report_creation_and_attributes(self):
        rep = pp.report(self.df)
        self.assertIsInstance(rep, PrepixReport)

        # Verify all expected attributes exist and are DataFrames
        self.assertIsInstance(rep.overview, pd.DataFrame)
        self.assertIsInstance(rep.issues, pd.DataFrame)
        self.assertIsInstance(rep.columns, pd.DataFrame)
        self.assertIsInstance(rep.missing, pd.DataFrame)
        self.assertIsInstance(rep.unique, pd.DataFrame)
        self.assertIsInstance(rep.numeric, pd.DataFrame)
        self.assertIsInstance(rep.duplicates, pd.DataFrame)

        # Overview contents
        overview_metrics = dict(zip(rep.overview["Metric"], rep.overview["Value"]))
        self.assertEqual(overview_metrics["Rows"], "6")
        self.assertEqual(overview_metrics["Columns"], "5")
        self.assertEqual(overview_metrics["Total Cells"], "30")
        self.assertEqual(overview_metrics["Columns with Missing"], "1")
        self.assertIn("Duplicate Rows", overview_metrics)

        # Issues detection
        issue_types = list(rep.issues["Issue"])
        self.assertTrue(
            any("Missing" in issue for issue in issue_types)
            or any("Outlier" in issue for issue in issue_types)
            or any("Zero Variance" in issue for issue in issue_types)
        )

    def test_empty_dataframe(self):
        rep_empty = pp.report(pd.DataFrame())
        self.assertIsInstance(rep_empty, PrepixReport)
        self.assertEqual(len(rep_empty.columns), 0)
        self.assertEqual(len(rep_empty.missing), 0)
        overview_dict = dict(zip(rep_empty.overview["Metric"], rep_empty.overview["Value"]))
        self.assertEqual(overview_dict["Rows"], "0")
        self.assertEqual(overview_dict["Columns"], "0")

    def test_to_dict_and_to_json(self):
        rep = pp.report(self.df)
        d = rep.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("overview", d)
        self.assertIn("issues", d)
        self.assertIn("columns", d)
        self.assertIn("missing", d)
        self.assertIn("unique", d)
        self.assertIn("numeric", d)
        self.assertIn("duplicates", d)

        # JSON export string
        json_str = rep.to_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed["overview"]), len(rep.overview))

    def test_str_and_repr(self):
        rep = pp.report(self.df)
        rep_str = str(rep)
        rep_repr = repr(rep)
        self.assertEqual(rep_str, rep_repr)
        self.assertIn("PREPIX QUALITY REPORT", rep_str)
        self.assertIn("DATASET OVERVIEW", rep_str)
        self.assertIn("KEY FINDINGS & ISSUES", rep_str)
        self.assertIn("RECOMMENDED ACTIONS", rep_str)

    def test_immutability(self):
        orig = self.df.copy(deep=True)
        _ = pp.report(self.df)
        pd.testing.assert_frame_equal(self.df, orig)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            pp.report([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
