"""
Prepix Example: Duplicate Analysis
Demonstrates duplicate_report() with full-row detection, subset detection,
different keep strategies, and direct extraction of duplicate records.
"""
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: DUPLICATE RECORD AUDIT")
    print("=" * 70)

    # Synthetic transactions dataset with exact and subset duplicates
    df = pd.DataFrame({
        "TransactionID": [1001, 1002, 1003, 1004, 1005, 1002, 1006],
        "CustomerID": ["C01", "C02", "C03", "C04", "C05", "C02", "C02"],
        "Amount": [50.0, 120.0, 30.0, 95.0, 200.0, 120.0, 75.0],
        "Store": ["NYC", "LA", "NYC", "Chicago", "LA", "LA", "LA"],
    })

    print("Input Dataset Preview (7 transactions):")
    print(df)

    # 1. Full-row duplicate audit (default keep='first')
    print("\n1. Full-Row Duplicate Summary (All columns identical):")
    full_dup = pp.duplicate_report(df)
    print(full_dup)

    # 2. Extract duplicate records directly (return_records=True)
    print("\n2. Extracted Duplicate Rows (return_records=True, keep='first'):")
    dup_records = pp.duplicate_report(df, return_records=True)
    print(dup_records)

    # 3. Duplicate check on a subset of columns (e.g. CustomerID)
    print("\n3. Subset Duplicate Check (subset=['CustomerID']):")
    subset_dup = pp.duplicate_report(df, subset=["CustomerID"])
    print(subset_dup)

    # 4. Impact of keep parameter ('first' vs False)
    print("\n4. Impact of keep parameter:")
    print("   keep='first': Only subsequent repeated rows count as duplicates.")
    print("   keep=False:   ALL occurrences of repeated rows are marked as duplicates.")

    rep_first = pp.duplicate_report(df, subset=["CustomerID"], keep="first")
    print(f"\nkeep='first' -> Duplicate Rows: {rep_first.iloc[0]['Duplicate Rows']}, Unique Rows: {rep_first.iloc[0]['Unique Rows']}")

    rep_all = pp.duplicate_report(df, subset=["CustomerID"], keep=False)
    print(f"keep=False   -> Duplicate Rows: {rep_all.iloc[0]['Duplicate Rows']}, Unique Rows: {rep_all.iloc[0]['Unique Rows']}")

    print("\nDuplicate analysis completed successfully!")


if __name__ == "__main__":
    main()
