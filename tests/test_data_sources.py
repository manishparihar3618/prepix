"""
Unit tests validating prepix functions across diverse DataFrame ingestion workflows:
- In-memory DataFrame (pd.DataFrame)
- CSV parsing (pd.read_csv)
- JSON parsing (pd.read_json)
- SQL queries (pd.read_sql via sqlite3)
"""
import io
import json
import sqlite3
import unittest
import pandas as pd
import prepix as pp


class TestDataSources(unittest.TestCase):
    def test_in_memory_dataframe(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [10.5, 20.0, None],
        })
        rep = pp.report(df)
        self.assertIsNotNone(rep)
        self.assertEqual(len(rep.overview), 12)
        self.assertEqual(len(rep.columns), 3)

    def test_csv_ingestion(self):
        csv_str = "col1,col2,col3\n10,foo,1.5\n20,bar,2.5\n30,foo,\n10,foo,1.5\n"
        df = pd.read_csv(io.StringIO(csv_str))
        self.assertEqual(df.shape, (4, 3))

        # Check missing report
        m = pp.missing_report(df)
        self.assertEqual(len(m), 3)
        self.assertEqual(m[m["Column"] == "col3"].iloc[0]["Missing Values"], 1)

        # Check duplicate report
        d = pp.duplicate_report(df)
        self.assertEqual(d.iloc[0]["Duplicate Rows"], 1)

    def test_json_ingestion(self):
        json_str = json.dumps([
            {"user": "alice", "score": 95, "active": True},
            {"user": "bob", "score": 80, "active": False},
            {"user": "charlie", "score": None, "active": True},
        ])
        df = pd.read_json(io.StringIO(json_str))
        self.assertEqual(df.shape, (3, 3))

        c = pp.column_summary(df)
        self.assertEqual(len(c), 3)

        n = pp.numeric_summary(df)
        self.assertEqual(len(n), 1)  # only 'score'
        self.assertEqual(n.iloc[0]["Count"], 2)

    def test_sqlite_ingestion(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE orders (order_id INT, amount REAL, customer TEXT)")
        cursor.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [
                (1, 100.0, "Acme"),
                (2, 250.5, "Beta"),
                (3, 75.25, "Gamma"),
            ],
        )
        conn.commit()

        df = pd.read_sql("SELECT * FROM orders", conn)
        conn.close()

        schema = {
            "order_id": {"type": "integer", "nullable": False, "unique": True},
            "amount": {"type": "float", "nullable": False},
            "customer": "string",
        }
        res = pp.schema_report(df, schema=schema)
        self.assertTrue((res["Status"] == "PASS").all())


if __name__ == "__main__":
    unittest.main()
