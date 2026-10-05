"""
Unit and regression tests for prepix.quality module.
"""
import unittest
import warnings
import numpy as np
import pandas as pd

from prepix.quality import column_summary, duplicate_report, missing_report


class TestQualityFunctions(unittest.TestCase):
    def setUp(self):
        # Dataset A - Normal dataset (no duplicates)
        self.df_a = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 75000.0, 90000.0, 110000.0],
            "department": ["HR", "IT", "IT", "Finance", "HR"],
            "is_active": [True, True, False, True, False]
        })

        # Dataset B - Missing values
        self.df_b_no_missing = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": ["x", "y", "z", "w"]
        })
        self.df_b_some_missing = pd.DataFrame({
            "col_0pct": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "col_10pct": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
            "col_30pct": [1, 2, 3, 4, 5, 6, 7, None, None, None],
            "col_50pct": [1, 2, 3, 4, 5, None, None, None, None, None],
            "col_100pct": [None] * 10
        })

        # Dataset C - Edge cases
        self.df_c_empty_no_cols = pd.DataFrame()
        self.df_c_empty_with_cols = pd.DataFrame(columns=["a", "b", "c"])
        self.df_c_single_row = pd.DataFrame({"a": [10], "b": ["hello"], "c": [np.nan]})
        self.df_c_single_col = pd.DataFrame({"single": [1, None, 3, None, 5]})
        self.df_c_all_missing = pd.DataFrame({"a": [None, None], "b": [np.nan, np.nan]})
        self.df_c_mixed_types = pd.DataFrame({
            "mixed_obj": [1, "two", 3.0, None],
            "mixed_num": [1, 2.5, 3, 4.0]
        })

        # Dataset D - Large-column dataset (120 columns)
        np.random.seed(42)
        data_d = {f"col_{i}": np.random.randn(50) for i in range(120)}
        data_d["col_with_na"] = [np.nan if i % 2 == 0 else 1.0 for i in range(50)]
        self.df_d_large_cols = pd.DataFrame(data_d)

        # Dataset E - Different data types
        self.df_e_types = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, np.nan, 4.4, 5.5],
            "str_obj_col": ["alpha", "beta", "gamma", None, "epsilon"],
            "str_dtype_col": pd.Series(["a", "b", None, "d", "e"], dtype="string"),
            "bool_col": [True, False, True, False, True],
            "nullable_bool_col": pd.Series([True, False, None, True, False], dtype="boolean"),
            "category_col": pd.Categorical(["cat", "dog", "cat", None, "bird"]),
            "datetime_col": pd.date_range("2025-01-01", periods=5, freq="D"),
            "datetime_with_na": [pd.Timestamp("2025-01-01"), None, pd.Timestamp("2025-01-03"), None, pd.Timestamp("2025-01-05")]
        })

        # Duplicate Test Datasets
        self.df_dup_one = pd.DataFrame({
            "id": [1, 2, 3, 2],
            "name": ["A", "B", "C", "B"]
        })
        self.df_dup_multiple = pd.DataFrame({
            "x": [1, 2, 1, 2, 3, 1],
            "y": ["a", "b", "a", "b", "c", "a"]
        })
        self.df_dup_all_subsequent = pd.DataFrame({
            "col1": [10, 10, 10, 10],
            "col2": ["same", "same", "same", "same"]
        })
        self.df_dup_with_nan = pd.DataFrame({
            "num": [1.0, np.nan, 2.0, np.nan],
            "cat": ["x", None, "y", None]
        })

    # ==========================================
    # FUNCTION TESTS: missing_report()
    # ==========================================

    def test_missing_report_dataset_a(self):
        rep = missing_report(self.df_a)
        self.assertIsInstance(rep, pd.DataFrame)
        expected_cols = ["Column", "Missing Values", "Missing Percentage(%)", "Status", "Suggestion"]
        self.assertListEqual(list(rep.columns), expected_cols)
        self.assertEqual(len(rep), 4)
        self.assertTrue((rep["Missing Values"] == 0).all())
        self.assertTrue((rep["Status"] == "No Missing").all())
        self.assertTrue((rep["Suggestion"] == "No Action Needed").all())

    def test_missing_report_dataset_b_thresholds(self):
        rep = missing_report(self.df_b_some_missing)
        self.assertEqual(len(rep), 5)

        row_100 = rep[rep["Column"] == "col_100pct"].iloc[0]
        self.assertEqual(row_100["Missing Values"], 10)
        self.assertEqual(row_100["Missing Percentage(%)"], 100.0)
        self.assertEqual(row_100["Status"], "High")
        self.assertEqual(row_100["Suggestion"], "Review Before Imputation")

        row_50 = rep[rep["Column"] == "col_50pct"].iloc[0]
        self.assertEqual(row_50["Missing Values"], 5)
        self.assertEqual(row_50["Missing Percentage(%)"], 50.0)
        self.assertEqual(row_50["Status"], "High")
        self.assertEqual(row_50["Suggestion"], "Review Before Imputation")

        row_30 = rep[rep["Column"] == "col_30pct"].iloc[0]
        self.assertEqual(row_30["Missing Values"], 3)
        self.assertEqual(row_30["Missing Percentage(%)"], 30.0)
        self.assertEqual(row_30["Status"], "Medium")
        self.assertEqual(row_30["Suggestion"], "Consider Median Imputation")

        row_10 = rep[rep["Column"] == "col_10pct"].iloc[0]
        self.assertEqual(row_10["Missing Values"], 1)
        self.assertEqual(row_10["Missing Percentage(%)"], 10.0)
        self.assertEqual(row_10["Status"], "Low")
        self.assertEqual(row_10["Suggestion"], "Consider Median Imputation")

        row_0 = rep[rep["Column"] == "col_0pct"].iloc[0]
        self.assertEqual(row_0["Missing Values"], 0)
        self.assertEqual(row_0["Missing Percentage(%)"], 0.0)
        self.assertEqual(row_0["Status"], "No Missing")
        self.assertEqual(row_0["Suggestion"], "No Action Needed")

    def test_missing_report_suggestions_by_dtype(self):
        rep = missing_report(self.df_e_types)
        row_str_obj = rep[rep["Column"] == "str_obj_col"].iloc[0]
        self.assertEqual(row_str_obj["Suggestion"], "Consider Mode Imputation")

        row_str_dtype = rep[rep["Column"] == "str_dtype_col"].iloc[0]
        self.assertEqual(row_str_dtype["Suggestion"], "Consider Mode Imputation")

        row_cat = rep[rep["Column"] == "category_col"].iloc[0]
        self.assertEqual(row_cat["Suggestion"], "Consider Mode Imputation")

        row_num = rep[rep["Column"] == "float_col"].iloc[0]
        self.assertEqual(row_num["Suggestion"], "Consider Median Imputation")

        row_dt = rep[rep["Column"] == "datetime_with_na"].iloc[0]
        self.assertEqual(row_dt["Suggestion"], "Review Before Imputation")

    def test_missing_report_no_warnings(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            _ = missing_report(self.df_e_types)
            self.assertEqual(len(recorded), 0, f"Unexpected warnings: {[str(w.message) for w in recorded]}")

    # ==========================================
    # FUNCTION TESTS: column_summary()
    # ==========================================

    def test_column_summary_dataset_a(self):
        summary = column_summary(self.df_a)
        self.assertIsInstance(summary, pd.DataFrame)
        expected_cols = [
            "Column",
            "Data Type",
            "Missing Values",
            "Missing Percentage(%)",
            "Unique Values",
            "Memory(Bytes)",
            "Suggested Type",
        ]
        self.assertListEqual(list(summary.columns), expected_cols)
        self.assertEqual(len(summary), 4)

        row_age = summary[summary["Column"] == "age"].iloc[0]
        self.assertEqual(row_age["Missing Values"], 0)
        self.assertEqual(row_age["Unique Values"], 5)
        self.assertEqual(row_age["Suggested Type"], "Numeric")

        row_bool = summary[summary["Column"] == "is_active"].iloc[0]
        self.assertEqual(row_bool["Suggested Type"], "Binary")

    def test_column_summary_large_dataset_d(self):
        summary = column_summary(self.df_d_large_cols)
        self.assertEqual(len(summary), 121)

    # ==========================================
    # FUNCTION TESTS: duplicate_report()
    # ==========================================

    def test_duplicate_report_no_duplicates(self):
        rep = duplicate_report(self.df_a)
        self.assertIsInstance(rep, pd.DataFrame)
        expected_cols = ["Total Rows", "Duplicate Rows", "Duplicate Percentage(%)", "Unique Rows", "Status", "Suggestion"]
        self.assertListEqual(list(rep.columns), expected_cols)
        self.assertEqual(len(rep), 1)

        row = rep.iloc[0]
        self.assertEqual(row["Total Rows"], 5)
        self.assertEqual(row["Duplicate Rows"], 0)
        self.assertEqual(row["Duplicate Percentage(%)"], 0.0)
        self.assertEqual(row["Unique Rows"], 5)
        self.assertEqual(row["Status"], "No Duplicates")
        self.assertEqual(row["Suggestion"], "No action required")

    def test_duplicate_report_one_duplicate(self):
        rep = duplicate_report(self.df_dup_one)
        row = rep.iloc[0]
        self.assertEqual(row["Total Rows"], 4)
        self.assertEqual(row["Duplicate Rows"], 1)
        self.assertEqual(row["Duplicate Percentage(%)"], 25.0)
        self.assertEqual(row["Unique Rows"], 3)
        self.assertEqual(row["Status"], "Duplicates Found")
        self.assertEqual(row["Suggestion"], "Consider reviewing and removing duplicate records")

    def test_duplicate_report_multiple_duplicates(self):
        rep = duplicate_report(self.df_dup_multiple)
        row = rep.iloc[0]
        self.assertEqual(row["Total Rows"], 6)
        # Unique: (1, 'a'), (2, 'b'), (3, 'c') -> 3 unique, 3 duplicates
        self.assertEqual(row["Duplicate Rows"], 3)
        self.assertEqual(row["Duplicate Percentage(%)"], 50.0)
        self.assertEqual(row["Unique Rows"], 3)
        self.assertEqual(row["Status"], "Duplicates Found")

    def test_duplicate_report_all_duplicates_subsequent(self):
        rep = duplicate_report(self.df_dup_all_subsequent)
        row = rep.iloc[0]
        self.assertEqual(row["Total Rows"], 4)
        self.assertEqual(row["Duplicate Rows"], 3)
        self.assertEqual(row["Duplicate Percentage(%)"], 75.0)
        self.assertEqual(row["Unique Rows"], 1)
        self.assertEqual(row["Status"], "Duplicates Found")

    def test_duplicate_report_empty_dataframe(self):
        # Empty with no columns
        rep0 = duplicate_report(self.df_c_empty_no_cols)
        self.assertEqual(len(rep0), 1)
        row0 = rep0.iloc[0]
        self.assertEqual(row0["Total Rows"], 0)
        self.assertEqual(row0["Duplicate Rows"], 0)
        self.assertEqual(row0["Duplicate Percentage(%)"], 0.0)
        self.assertEqual(row0["Unique Rows"], 0)
        self.assertEqual(row0["Status"], "No Duplicates")
        self.assertEqual(row0["Suggestion"], "No action required")

        # Empty with columns
        rep_cols = duplicate_report(self.df_c_empty_with_cols)
        self.assertEqual(len(rep_cols), 1)
        row_c = rep_cols.iloc[0]
        self.assertEqual(row_c["Total Rows"], 0)
        self.assertEqual(row_c["Duplicate Rows"], 0)
        self.assertEqual(row_c["Duplicate Percentage(%)"], 0.0)
        self.assertEqual(row_c["Unique Rows"], 0)
        self.assertEqual(row_c["Status"], "No Duplicates")

    def test_duplicate_report_with_missing_values(self):
        rep = duplicate_report(self.df_dup_with_nan)
        row = rep.iloc[0]
        # (np.nan, None) occurs twice, so 1 duplicate row
        self.assertEqual(row["Total Rows"], 4)
        self.assertEqual(row["Duplicate Rows"], 1)
        self.assertEqual(row["Duplicate Percentage(%)"], 25.0)
        self.assertEqual(row["Unique Rows"], 3)
        self.assertEqual(row["Status"], "Duplicates Found")

    def test_duplicate_report_mixed_types(self):
        rep = duplicate_report(self.df_c_mixed_types)
        row = rep.iloc[0]
        self.assertEqual(row["Total Rows"], 4)
        self.assertEqual(row["Duplicate Rows"], 0)
        self.assertEqual(row["Unique Rows"], 4)
        self.assertEqual(row["Status"], "No Duplicates")

    def test_duplicate_report_large_columns(self):
        rep = duplicate_report(self.df_d_large_cols)
        row = rep.iloc[0]
        self.assertEqual(row["Total Rows"], 50)
        self.assertEqual(row["Duplicate Rows"], 0)
        self.assertEqual(row["Unique Rows"], 50)

    # ==========================================
    # EDGE CASE TESTS (Dataset C)
    # ==========================================

    def test_empty_dataframes(self):
        # Empty DataFrame with no columns
        rep0 = missing_report(self.df_c_empty_no_cols)
        self.assertIsInstance(rep0, pd.DataFrame)
        self.assertEqual(len(rep0), 0)
        self.assertListEqual(list(rep0.columns), ["Column", "Missing Values", "Missing Percentage(%)", "Status", "Suggestion"])

        summ0 = column_summary(self.df_c_empty_no_cols)
        self.assertIsInstance(summ0, pd.DataFrame)
        self.assertEqual(len(summ0), 0)
        self.assertListEqual(list(summ0.columns), ["Column", "Data Type", "Missing Values", "Missing Percentage(%)", "Unique Values", "Memory(Bytes)", "Suggested Type"])

        # Empty DataFrame with columns
        rep_cols = missing_report(self.df_c_empty_with_cols)
        self.assertEqual(len(rep_cols), 3)
        self.assertTrue((rep_cols["Missing Values"] == 0).all())
        self.assertTrue((rep_cols["Missing Percentage(%)"] == 0.0).all())

        summ_cols = column_summary(self.df_c_empty_with_cols)
        self.assertEqual(len(summ_cols), 3)
        self.assertTrue((summ_cols["Missing Values"] == 0).all())

    def test_single_row_and_column(self):
        rep_row = missing_report(self.df_c_single_row)
        self.assertEqual(len(rep_row), 3)
        summ_row = column_summary(self.df_c_single_row)
        self.assertEqual(len(summ_row), 3)

        rep_col = missing_report(self.df_c_single_col)
        self.assertEqual(len(rep_col), 1)
        summ_col = column_summary(self.df_c_single_col)
        self.assertEqual(len(summ_col), 1)

    def test_all_missing(self):
        rep = missing_report(self.df_c_all_missing)
        self.assertTrue((rep["Missing Percentage(%)"] == 100.0).all())
        summ = column_summary(self.df_c_all_missing)
        self.assertTrue((summ["Missing Percentage(%)"] == 100.0).all())

    def test_duplicate_columns(self):
        df_dup = pd.DataFrame([[1, "a"], [None, "b"], [3, "c"]], columns=["x", "x"])
        rep = missing_report(df_dup)
        self.assertEqual(len(rep), 2)
        summ = column_summary(df_dup)
        self.assertEqual(len(summ), 2)

    # ==========================================
    # INVALID INPUT TESTS
    # ==========================================

    def test_invalid_inputs(self):
        invalid_inputs = [None, [], "data", 123, {"a": [1, 2]}, (1, 2)]
        for inp in invalid_inputs:
            with self.subTest(inp=inp):
                with self.assertRaises(TypeError):
                    missing_report(inp)
                with self.assertRaises(TypeError):
                    column_summary(inp)
                with self.assertRaises(TypeError):
                    duplicate_report(inp)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            missing_report(self.df_a, low_threshold=-5)
        with self.assertRaises(ValueError):
            missing_report(self.df_a, low_threshold=105)
        with self.assertRaises(ValueError):
            missing_report(self.df_a, low_threshold=60, medium_threshold=20)
        with self.assertRaises(TypeError):
            missing_report(self.df_a, low_threshold="10")
        with self.assertRaises(TypeError):
            missing_report(self.df_a, low_threshold=True)
        with self.assertRaises(TypeError):
            missing_report(self.df_a, medium_threshold=False)

    # ==========================================
    # ORIGINAL DATAFRAME SAFETY / IMMUTABILITY
    # ==========================================

    def test_dataframe_immutability(self):
        test_dfs = [
            self.df_a,
            self.df_b_some_missing,
            self.df_c_single_row,
            self.df_c_single_col,
            self.df_d_large_cols,
            self.df_e_types,
            self.df_dup_multiple,
        ]
        for i, df in enumerate(test_dfs):
            with self.subTest(df_index=i):
                orig_copy1 = df.copy(deep=True)
                _ = missing_report(df)
                pd.testing.assert_frame_equal(df, orig_copy1)

                orig_copy2 = df.copy(deep=True)
                _ = column_summary(df)
                pd.testing.assert_frame_equal(df, orig_copy2)

                orig_copy3 = df.copy(deep=True)
                _ = duplicate_report(df)
                pd.testing.assert_frame_equal(df, orig_copy3)


if __name__ == "__main__":
    unittest.main()
