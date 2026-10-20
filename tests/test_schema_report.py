"""
Unit and regression tests for schema_report() in prepix.schema.
"""
from __future__ import annotations

import datetime
import unittest
import numpy as np
import pandas as pd

import prepix as pp
from prepix import schema_report


class TestSchemaReport(unittest.TestCase):
    def setUp(self):
        self.df_valid = pd.DataFrame({
            "Name": ["Aman", "Priya", "Rahul"],
            "Age": [25, 30, 28],
            "Salary": [50000.0, 60000.0, 55000.0],
            "IsActive": [True, False, True],
        })

    def test_full_schema_match_simple(self):
        schema = {
            "Name": "string",
            "Age": "numeric",
            "Salary": "float",
            "IsActive": "boolean",
        }
        res = schema_report(self.df_valid, schema=schema)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(
            list(res.columns),
            ["Column", "Check", "Expected", "Actual", "Status", "Message"],
        )
        self.assertTrue((res["Status"] == "PASS").all())
        self.assertEqual(len(res), 8)  # 4 column_presence + 4 data_type checks

    def test_full_schema_match_python_types(self):
        schema = {
            "Name": str,
            "Age": int,
            "Salary": float,
            "IsActive": bool,
        }
        res = schema_report(self.df_valid, schema=schema)
        self.assertTrue((res["Status"] == "PASS").all())

    def test_missing_required_column(self):
        schema = {
            "Name": "string",
            "NonExistent": "numeric",
        }
        res = schema_report(self.df_valid, schema=schema)
        fail_rows = res[res["Status"] == "FAIL"]
        self.assertEqual(len(fail_rows), 1)

        row = fail_rows.iloc[0]
        self.assertEqual(row["Column"], "NonExistent")
        self.assertEqual(row["Check"], "column_presence")
        self.assertEqual(row["Expected"], "present")
        self.assertEqual(row["Actual"], "missing")
        self.assertIn("missing from DataFrame", row["Message"])

    def test_optional_missing_column(self):
        schema = {
            "Name": "string",
            "OptionalCol": {"type": "numeric", "required": False},
        }
        res = schema_report(self.df_valid, schema=schema)
        opt_rows = res[res["Column"] == "OptionalCol"]
        self.assertEqual(len(opt_rows), 1)
        row = opt_rows.iloc[0]
        self.assertEqual(row["Check"], "column_presence")
        self.assertEqual(row["Expected"], "optional")
        self.assertEqual(row["Actual"], "missing")
        self.assertEqual(row["Status"], "PASS")

    def test_unexpected_column_allowed(self):
        schema = {
            "Name": "string",
            "Age": "numeric",
        }
        # Salary and IsActive are extra/unexpected
        res = schema_report(self.df_valid, schema=schema, allow_unexpected=True)
        unexpected_rows = res[res["Check"] == "unexpected_column"]
        self.assertEqual(len(unexpected_rows), 2)
        self.assertTrue((unexpected_rows["Status"] == "PASS").all())
        self.assertTrue((unexpected_rows["Expected"] == "allowed").all())
        self.assertTrue((unexpected_rows["Actual"] == "present").all())

    def test_unexpected_column_disallowed(self):
        schema = {
            "Name": "string",
            "Age": "numeric",
        }
        res = schema_report(self.df_valid, schema=schema, allow_unexpected=False)
        unexpected_rows = res[res["Check"] == "unexpected_column"]
        self.assertEqual(len(unexpected_rows), 2)
        self.assertTrue((unexpected_rows["Status"] == "FAIL").all())
        self.assertTrue((unexpected_rows["Expected"] == "absent").all())
        self.assertTrue((unexpected_rows["Actual"] == "present").all())

    def test_incorrect_data_type(self):
        df = pd.DataFrame({"Age": ["not_a_number", "twenty"]})
        schema = {"Age": "numeric"}
        res = schema_report(df, schema=schema)

        type_check = res[res["Check"] == "data_type"].iloc[0]
        self.assertEqual(type_check["Status"], "FAIL")
        self.assertEqual(type_check["Expected"], "numeric")
        self.assertIn("Expected data type 'numeric'", type_check["Message"])

    def test_missing_values_in_required_non_null_column(self):
        df = pd.DataFrame({
            "col_with_na": [1, None, 3],
            "col_clean": [10, 20, 30],
        })
        schema = {
            "col_with_na": {"type": "numeric", "nullable": False},
            "col_clean": {"type": "numeric", "nullable": False},
        }
        res = schema_report(df, schema=schema)

        fail_row = res[(res["Column"] == "col_with_na") & (res["Check"] == "non_null")].iloc[0]
        self.assertEqual(fail_row["Status"], "FAIL")
        self.assertEqual(fail_row["Expected"], "no_missing")
        self.assertEqual(fail_row["Actual"], "1 missing")
        self.assertIn("contains 1 missing value(s) in non-nullable column", fail_row["Message"])

        pass_row = res[(res["Column"] == "col_clean") & (res["Check"] == "non_null")].iloc[0]
        self.assertEqual(pass_row["Status"], "PASS")
        self.assertEqual(pass_row["Expected"], "no_missing")
        self.assertEqual(pass_row["Actual"], "0 missing")

    def test_non_null_alias_key(self):
        df = pd.DataFrame({"val": [1, None, 2]})
        schema = {"val": {"type": "numeric", "non_null": True}}
        res = schema_report(df, schema=schema)
        row = res[res["Check"] == "non_null"].iloc[0]
        self.assertEqual(row["Status"], "FAIL")

    def test_nullable_allowed(self):
        df = pd.DataFrame({"val": [1, None, 2]})
        schema = {"val": {"type": "numeric", "nullable": True}}
        res = schema_report(df, schema=schema)
        row = res[res["Check"] == "non_null"].iloc[0]
        self.assertEqual(row["Status"], "PASS")
        self.assertEqual(row["Expected"], "nullable")
        self.assertEqual(row["Actual"], "1 missing")

    def test_duplicate_values_in_column_that_must_be_unique(self):
        df_dups = pd.DataFrame({"id": [101, 101, 102]})
        schema = {"id": {"type": "integer", "unique": True}}
        res = schema_report(df_dups, schema=schema)

        dup_row = res[res["Check"] == "unique"].iloc[0]
        self.assertEqual(dup_row["Status"], "FAIL")
        self.assertEqual(dup_row["Expected"], "unique")
        self.assertEqual(dup_row["Actual"], "1 duplicate(s)")
        self.assertIn("contains 1 duplicate value(s) in unique column", dup_row["Message"])

        # Test unique passes
        df_unique = pd.DataFrame({"id": [101, 102, 103]})
        res_unique = schema_report(df_unique, schema=schema)
        pass_row = res_unique[res_unique["Check"] == "unique"].iloc[0]
        self.assertEqual(pass_row["Status"], "PASS")
        self.assertEqual(pass_row["Expected"], "unique")
        self.assertEqual(pass_row["Actual"], "0 duplicates")

    def test_empty_dataframes_0x0(self):
        df_empty = pd.DataFrame()
        schema = {"A": "integer", "B": "string"}
        res = schema_report(df_empty, schema=schema)
        self.assertEqual(len(res), 2)
        self.assertTrue((res["Status"] == "FAIL").all())
        self.assertTrue((res["Actual"] == "missing").all())

    def test_empty_dataframes_0xN_with_columns(self):
        df_empty_cols = pd.DataFrame(columns=["A", "B"])
        df_empty_cols["A"] = df_empty_cols["A"].astype(int)
        df_empty_cols["B"] = df_empty_cols["B"].astype(str)

        schema = {
            "A": {"type": "integer", "nullable": False, "unique": True},
            "B": "string",
        }
        res = schema_report(df_empty_cols, schema=schema)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue((res["Status"] == "PASS").all())

    def test_nullable_pandas_data_types(self):
        df_nullable = pd.DataFrame({
            "int_col": pd.Series([1, 2, None], dtype="Int64"),
            "float_col": pd.Series([1.5, None, 3.5], dtype="Float64"),
            "str_col": pd.Series(["hello", "world", None], dtype="string"),
            "bool_col": pd.Series([True, False, None], dtype="boolean"),
        })
        schema = {
            "int_col": "integer",
            "float_col": "float",
            "str_col": "string",
            "bool_col": "boolean",
        }
        res = schema_report(df_nullable, schema=schema)
        type_checks = res[res["Check"] == "data_type"]
        self.assertEqual(len(type_checks), 4)
        self.assertTrue((type_checks["Status"] == "PASS").all())

    def test_invalid_schema_definitions(self):
        # Non-dict schema
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema="invalid_schema")
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema=["Name", "Age"])

        # Empty schema
        with self.assertRaises(ValueError):
            schema_report(self.df_valid, schema={})

        # Non-DataFrame input
        with self.assertRaises(TypeError):
            schema_report([1, 2, 3], schema={"a": "int"})

        # Non-boolean allow_unexpected
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema={"Name": "string"}, allow_unexpected="True")

        # Invalid spec type
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema={"Name": [1, 2, 3]})

        # Invalid key in dict spec
        with self.assertRaises(ValueError):
            schema_report(self.df_valid, schema={"Name": {"type": "string", "invalid_key": True}})

        # Invalid value type in dict spec
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema={"Name": {"type": "string", "nullable": "yes"}})
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema={"Name": {"type": "string", "unique": 1}})
        with self.assertRaises(TypeError):
            schema_report(self.df_valid, schema={"Name": {"type": "string", "required": "no"}})

        # Unknown data type
        with self.assertRaises(ValueError):
            schema_report(self.df_valid, schema={"Name": "unknown_fantasy_type"})

    def test_duplicate_column_names(self):
        df_dups = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "A"])
        schema = {"A": "integer"}
        res = schema_report(df_dups, schema=schema)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue((res["Status"] == "PASS").all())

    def test_original_dataframe_remaining_unchanged(self):
        orig = self.df_valid.copy(deep=True)
        schema = {
            "Name": "string",
            "Age": {"type": "numeric", "nullable": False, "unique": True},
        }
        _ = schema_report(self.df_valid, schema=schema)
        pd.testing.assert_frame_equal(self.df_valid, orig)

    def test_datetime_and_category_types(self):
        df = pd.DataFrame({
            "dates": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "cats": pd.Categorical(["cat", "dog"]),
            "deltas": pd.to_timedelta([1, 2], unit="D"),
        })
        schema = {
            "dates": "datetime",
            "cats": "category",
            "deltas": "timedelta",
        }
        res = schema_report(df, schema=schema)
        type_checks = res[res["Check"] == "data_type"]
        self.assertEqual(len(type_checks), 3)
        self.assertTrue((type_checks["Status"] == "PASS").all())

    def test_public_api_import_consistency(self):
        # Must be accessible directly from pp.schema_report
        self.assertTrue(callable(pp.schema_report))
        self.assertIs(pp.schema_report, schema_report)


if __name__ == "__main__":
    unittest.main()
