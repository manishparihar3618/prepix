"""
Comprehensive test suite for prepix across diverse dataset sizes, data types, and structural patterns.
Includes:
- Small (0x0, 0xN, Nx0, 1x1), medium, and large (100k+ rows, 250+ cols)
- Diverse dtypes (int8..64, uint8..64, float16..64, nullable types, complex, bool, string, categorical, datetime, timedelta, period, unhashable objects)
- Edge cases (NaNs, Infs, duplicates, constant columns, MultiIndex, non-string headers, emojis)
"""
import unittest
import numpy as np
import pandas as pd

from prepix import (
    column_summary,
    duplicate_report,
    missing_report,
    numeric_summary,
    unique_report,
)


class TestDatasetVarieties(unittest.TestCase):
    # =========================================================================
    # 1. DATASET SIZE VARIETIES (Small, Medium, Large, Wide, Tall)
    # =========================================================================

    def test_size_empty_zero_by_zero(self):
        df = pd.DataFrame()
        self.assertEqual(len(missing_report(df)), 0)
        self.assertEqual(len(column_summary(df)), 0)
        self.assertEqual(len(duplicate_report(df)), 1)
        self.assertEqual(len(unique_report(df)), 0)

    def test_size_zero_rows_multi_columns(self):
        df = pd.DataFrame(columns=["col_a", "col_b", "col_c", "col_d"])
        self.assertEqual(len(missing_report(df)), 4)
        self.assertEqual(len(column_summary(df)), 4)
        dup = duplicate_report(df)
        self.assertEqual(dup.iloc[0]["Total Rows"], 0)
        self.assertEqual(dup.iloc[0]["Duplicate Rows"], 0)
        self.assertEqual(len(unique_report(df)), 4)

    def test_size_multi_rows_zero_columns(self):
        df = pd.DataFrame(index=range(10))
        self.assertEqual(len(missing_report(df)), 0)
        self.assertEqual(len(column_summary(df)), 0)
        dup = duplicate_report(df)
        self.assertEqual(dup.iloc[0]["Total Rows"], 10)
        self.assertEqual(len(unique_report(df)), 0)

    def test_size_single_cell_one_by_one(self):
        df = pd.DataFrame({"only_val": [42]})
        self.assertEqual(len(missing_report(df)), 1)
        self.assertEqual(len(column_summary(df)), 1)
        self.assertEqual(len(duplicate_report(df)), 1)
        self.assertEqual(len(unique_report(df)), 1)

    def test_size_single_column_many_rows(self):
        df = pd.DataFrame({"tall_col": list(range(1000))})
        self.assertEqual(len(missing_report(df)), 1)
        self.assertEqual(len(column_summary(df)), 1)
        dup = duplicate_report(df)
        self.assertEqual(dup.iloc[0]["Total Rows"], 1000)
        self.assertEqual(dup.iloc[0]["Duplicate Rows"], 0)
        u_rep = unique_report(df)
        self.assertEqual(u_rep.iloc[0]["Unique Values"], 1000)

    def test_size_single_row_many_columns(self):
        data = {f"c_{i}": [i] for i in range(100)}
        df = pd.DataFrame(data)
        self.assertEqual(len(missing_report(df)), 100)
        self.assertEqual(len(column_summary(df)), 100)
        self.assertEqual(len(duplicate_report(df)), 1)
        self.assertEqual(len(unique_report(df)), 100)

    def test_size_large_tall_dataset(self):
        # 100,000 rows
        n = 100_000
        df = pd.DataFrame({
            "id": np.arange(n),
            "metric_1": np.random.randn(n),
            "category": np.random.choice(["X", "Y", "Z"], size=n),
            "flag": np.random.choice([True, False], size=n),
        })
        df.loc[::10, "metric_1"] = np.nan

        m_rep = missing_report(df)
        self.assertEqual(len(m_rep), 4)
        c_rep = column_summary(df)
        self.assertEqual(len(c_rep), 4)
        d_rep = duplicate_report(df)
        self.assertEqual(d_rep.iloc[0]["Total Rows"], n)
        u_rep = unique_report(df)
        self.assertEqual(len(u_rep), 4)

    def test_size_large_wide_dataset(self):
        # 250 columns
        n_cols = 250
        data = {f"feat_{i}": np.random.randint(0, 10, size=50) for i in range(n_cols)}
        df = pd.DataFrame(data)
        self.assertEqual(len(missing_report(df)), n_cols)
        self.assertEqual(len(column_summary(df)), n_cols)
        self.assertEqual(len(unique_report(df)), n_cols)
        self.assertEqual(len(duplicate_report(df)), 1)

    # =========================================================================
    # 2. DATA TYPE VARIETIES (Numeric, Datetime, String, Bool, Complex, Objects)
    # =========================================================================

    def test_dtypes_standard_integers(self):
        df = pd.DataFrame({
            "i8": np.array([1, -2, 3], dtype=np.int8),
            "i16": np.array([100, -200, 300], dtype=np.int16),
            "i32": np.array([10000, -20000, 30000], dtype=np.int32),
            "i64": np.array([100000, -200000, 300000], dtype=np.int64),
            "u8": np.array([1, 2, 3], dtype=np.uint8),
            "u16": np.array([100, 200, 300], dtype=np.uint16),
            "u32": np.array([10000, 20000, 30000], dtype=np.uint32),
            "u64": np.array([100000, 200000, 300000], dtype=np.uint64),
        })
        summ = column_summary(df)
        self.assertEqual(len(summ), 8)
        self.assertTrue((summ["Suggested Type"] == "Numeric").all())

    def test_dtypes_nullable_extension_types(self):
        df = pd.DataFrame({
            "null_i8": pd.Series([1, None, 3], dtype="Int8"),
            "null_i64": pd.Series([100, 200, None], dtype="Int64"),
            "null_u64": pd.Series([10, None, 30], dtype="UInt64"),
            "null_f32": pd.Series([1.5, None, 3.5], dtype="Float32"),
            "null_f64": pd.Series([None, 2.5, 3.5], dtype="Float64"),
            "null_bool": pd.Series([True, False, None], dtype="boolean"),
            "null_str": pd.Series(["hello", None, "world"], dtype="string"),
        })
        m_rep = missing_report(df)
        self.assertEqual(len(m_rep), 7)
        self.assertTrue((m_rep["Missing Values"] == 1).all())

        c_rep = column_summary(df)
        self.assertEqual(len(c_rep), 7)
        type_dict = dict(zip(c_rep["Column"], c_rep["Suggested Type"]))
        self.assertEqual(type_dict["null_i8"], "Numeric")
        self.assertEqual(type_dict["null_f64"], "Numeric")
        self.assertEqual(type_dict["null_bool"], "Binary")

        u_rep = unique_report(df, dropna=True)
        self.assertTrue((u_rep["Unique Values"] == 2).all())

    def test_dtypes_floats_and_special_values(self):
        df = pd.DataFrame({
            "f16": np.array([1.5, 2.5, np.nan], dtype=np.float16),
            "f32": np.array([1.5, np.nan, 3.5], dtype=np.float32),
            "f64": np.array([np.nan, 2.5, 3.5], dtype=np.float64),
            "infinities": [np.inf, -np.inf, 0.0],
            "subnormals": [1e-308, -1e-308, 0.0],
            "large_floats": [1e308, -1e308, 1e200],
        })
        self.assertEqual(len(missing_report(df)), 6)
        self.assertEqual(len(column_summary(df)), 6)
        self.assertEqual(len(unique_report(df)), 6)

    def test_dtypes_complex_numbers(self):
        df = pd.DataFrame({
            "c64": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
            "c128": np.array([1-2j, 3-4j, 5-6j], dtype=np.complex128),
        })
        summ = column_summary(df)
        self.assertEqual(len(summ), 2)
        self.assertEqual(len(unique_report(df)), 2)

    def test_dtypes_datetime_variants(self):
        df = pd.DataFrame({
            "dt_naive": pd.date_range("2025-01-01", periods=4, freq="D"),
            "dt_tz": pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
            "dt_with_nat": [pd.Timestamp("2025-01-01"), pd.NaT, pd.Timestamp("2025-01-03"), pd.NaT],
            "dt_microsec": pd.date_range("2025-01-01 00:00:00.000001", periods=4, freq="us"),
        })
        m_rep = missing_report(df)
        self.assertEqual(m_rep[m_rep["Column"] == "dt_with_nat"].iloc[0]["Missing Values"], 2)

        summ = column_summary(df)
        self.assertEqual(summ[summ["Column"] == "dt_naive"].iloc[0]["Suggested Type"], "Datetime")
        self.assertEqual(summ[summ["Column"] == "dt_tz"].iloc[0]["Suggested Type"], "Datetime")

    def test_dtypes_timedelta_and_period(self):
        df = pd.DataFrame({
            "td_col": pd.to_timedelta([1, 2, 3, 4], unit="D"),
            "td_with_nat": pd.to_timedelta([1, None, 3, None], unit="s"),
            "period_col": pd.period_range("2025-01", periods=4, freq="M"),
        })
        self.assertEqual(len(missing_report(df)), 3)
        self.assertEqual(len(column_summary(df)), 3)
        self.assertEqual(len(unique_report(df)), 3)

    def test_dtypes_categoricals(self):
        df = pd.DataFrame({
            "cat_str": pd.Categorical(["apple", "banana", "apple", "cherry"]),
            "cat_num": pd.Categorical([1, 2, 1, 3]),
            "cat_ordered": pd.Categorical(["low", "med", "high", "low"], ordered=True),
            "cat_unused": pd.Categorical(["a", "b", "a", "b"], categories=["a", "b", "c", "d"]),
            "cat_with_nan": pd.Categorical(["a", None, "b", None]),
        })
        self.assertEqual(len(missing_report(df)), 5)
        summ = column_summary(df)
        self.assertEqual(len(summ), 5)
        self.assertEqual(len(unique_report(df)), 5)

    def test_dtypes_unhashable_and_nested_objects(self):
        class SampleObj:
            def __init__(self, val):
                self.val = val
            def __repr__(self):
                return f"SampleObj({self.val})"

        df = pd.DataFrame({
            "list_col": [[1, 2], [3, 4], [1, 2], None],
            "dict_col": [{"k": 1}, {"k": 2}, {"k": 1}, {"k": 3}],
            "obj_col": [SampleObj(10), SampleObj(20), SampleObj(10), SampleObj(30)],
        })
        m_rep = missing_report(df)
        self.assertEqual(len(m_rep), 3)
        self.assertEqual(m_rep[m_rep["Column"] == "list_col"].iloc[0]["Missing Values"], 1)

        c_rep = column_summary(df)
        self.assertEqual(len(c_rep), 3)

        d_rep = duplicate_report(df)
        self.assertEqual(len(d_rep), 1)

        u_rep = unique_report(df)
        self.assertEqual(len(u_rep), 3)

    # =========================================================================
    # 3. DISTRIBUTION & CARDINALITY PATTERNS
    # =========================================================================

    def test_distribution_all_missing_dataframe(self):
        df = pd.DataFrame({
            "col1": [np.nan, None, np.nan],
            "col2": [None, None, None],
        })
        m = missing_report(df)
        self.assertTrue((m["Missing Percentage(%)"] == 100.0).all())
        self.assertTrue((m["Status"] == "High").all())

    def test_distribution_all_duplicates(self):
        df = pd.DataFrame({
            "x": [1, 1, 1, 1, 1],
            "y": ["a", "a", "a", "a", "a"],
        })
        dup = duplicate_report(df)
        self.assertEqual(dup.iloc[0]["Total Rows"], 5)
        self.assertEqual(dup.iloc[0]["Duplicate Rows"], 4)
        self.assertEqual(dup.iloc[0]["Unique Rows"], 1)
        self.assertEqual(dup.iloc[0]["Duplicate Percentage(%)"], 80.0)

    def test_distribution_cardinality_tiers(self):
        n = 100
        df = pd.DataFrame({
            "constant": [10] * n,                                    # 1 unique -> Constant
            "unique_id": list(range(n)),                             # 100 unique -> Very High (Unique)
            "near_unique": list(range(95)) + [0, 1, 2, 3, 4],        # 95 unique -> Very High
            "high_div": [i % 60 for i in range(n)],                  # 60 unique -> High
            "med_div": [i % 25 for i in range(n)],                   # 25 unique -> Medium
            "low_div": [i % 5 for i in range(n)],                    # 5 unique -> Low
            "binary": [i % 2 for i in range(n)],                     # 2 unique -> Low (Binary)
        })
        u = unique_report(df)
        card_map = dict(zip(u["Column"], u["Cardinality"]))
        self.assertEqual(card_map["constant"], "Constant")
        self.assertEqual(card_map["unique_id"], "Very High (Unique)")
        self.assertEqual(card_map["near_unique"], "Very High")
        self.assertEqual(card_map["high_div"], "High")
        self.assertEqual(card_map["med_div"], "Medium")
        self.assertEqual(card_map["low_div"], "Low")

    # =========================================================================
    # 4. STRUCTURAL, HEADER, AND INDEXING EDGE CASES
    # =========================================================================

    def test_structure_numeric_and_tuple_headers(self):
        df_int_cols = pd.DataFrame([[1, 2], [3, 4]], columns=[101, 202])
        self.assertEqual(len(missing_report(df_int_cols)), 2)
        self.assertEqual(len(column_summary(df_int_cols)), 2)
        self.assertEqual(len(unique_report(df_int_cols)), 2)

        df_multi_cols = pd.DataFrame([[1, 2], [3, 4]], columns=pd.MultiIndex.from_tuples([("A", "x"), ("B", "y")]))
        self.assertEqual(len(missing_report(df_multi_cols)), 2)
        self.assertEqual(len(column_summary(df_multi_cols)), 2)
        self.assertEqual(len(unique_report(df_multi_cols)), 2)

    def test_structure_duplicate_column_names(self):
        df_dups = pd.DataFrame([[1, "a", None], [2, "b", 3]], columns=["val", "val", "other"])
        self.assertEqual(len(missing_report(df_dups)), 3)
        self.assertEqual(len(column_summary(df_dups)), 3)
        self.assertEqual(len(unique_report(df_dups)), 3)

    def test_structure_special_column_names_and_unicode(self):
        df_unicode = pd.DataFrame({
            "col with spaces": [1, 2],
            "emoji_🚀_🌟": [3, 4],
            "special_!@#$%^&*()": [5, 6],
            "": [7, 8],
            "   leading_trailing   ": [9, 10],
            "newline\n\tcol": [11, 12],
        })
        self.assertEqual(len(missing_report(df_unicode)), 6)
        self.assertEqual(len(column_summary(df_unicode)), 6)
        self.assertEqual(len(unique_report(df_unicode)), 6)

    def test_structure_custom_and_multiindex_rows(self):
        df = pd.DataFrame(
            {"a": [10, 20, 30], "b": ["x", "y", "z"]},
            index=pd.MultiIndex.from_tuples([("g1", 1), ("g1", 2), ("g2", 1)], names=["grp", "sub"]),
        )
        self.assertEqual(len(missing_report(df)), 2)
        self.assertEqual(len(column_summary(df)), 2)
        dup = duplicate_report(df)
        self.assertEqual(dup.iloc[0]["Total Rows"], 3)
        self.assertEqual(len(unique_report(df)), 2)

    # =========================================================================
    # 5. IMMUTABILITY AND RETURN TYPES ACROSS ALL VARIETIES
    # =========================================================================

    def test_immutability_guarantee(self):
        datasets = [
            pd.DataFrame(),
            pd.DataFrame({"x": [1, 2, None]}),
            pd.DataFrame({"str": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]}),
            pd.DataFrame([[1, 2], [3, 4]], columns=["same", "same"]),
            pd.DataFrame({"dt": pd.date_range("2025-01-01", periods=3)}),
        ]
        for idx, df in enumerate(datasets):
            with self.subTest(dataset_idx=idx):
                copy1 = df.copy(deep=True)
                _ = missing_report(df)
                pd.testing.assert_frame_equal(df, copy1)

                copy2 = df.copy(deep=True)
                _ = column_summary(df)
                pd.testing.assert_frame_equal(df, copy2)

                copy3 = df.copy(deep=True)
                _ = duplicate_report(df)
                pd.testing.assert_frame_equal(df, copy3)

                copy4 = df.copy(deep=True)
                _ = unique_report(df)
                pd.testing.assert_frame_equal(df, copy4)

                copy5 = df.copy(deep=True)
                _ = numeric_summary(df)
                pd.testing.assert_frame_equal(df, copy5)


if __name__ == "__main__":
    unittest.main()
