"""
Prepix Example: Schema Validation & Contract Verification
Demonstrates schema_report() with passing contracts and intentional validation failures.
"""
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: SCHEMA CONTRACT VALIDATION")
    print("=" * 70)

    # Define a business schema contract
    contract = {
        "AccountID": {"type": "integer", "nullable": False, "unique": True},
        "AccountHolder": "string",
        "Balance": {"type": "numeric", "nullable": False},
        "Tier": "string",
    }

    # 1. Dataset A: Clean and fully compliant
    print("\n--- SCENARIO 1: FULLY COMPLIANT DATASET ---")
    valid_df = pd.DataFrame({
        "AccountID": [1001, 1002, 1003],
        "AccountHolder": ["Alice Smith", "Bob Jones", "Carol White"],
        "Balance": [2500.50, 10400.00, 310.25],
        "Tier": ["Gold", "Platinum", "Silver"],
    })

    valid_report = pp.schema_report(valid_df, schema=contract, allow_unexpected=True)
    print(valid_report[["Column", "Check", "Expected", "Actual", "Status", "Message"]])
    passes = (valid_report["Status"] == "PASS").all()
    print(f"\nAll checks passed? -> {passes}")

    # 2. Dataset B: Non-compliant with multiple intentional violations:
    #    - AccountID has duplicate (1002 repeated)
    #    - Balance contains a missing value (None)
    #    - Missing required column 'Tier'
    #    - Unexpected column 'InternalNotes' when allow_unexpected=False
    print("\n--- SCENARIO 2: DATASET WITH INTENTIONAL VIOLATIONS ---")
    invalid_df = pd.DataFrame({
        "AccountID": [1001, 1002, 1002],  # Duplicate violation
        "AccountHolder": ["Alice", "Bob", "Charlie"],
        "Balance": [1500.0, None, 450.0],  # Missing value in non-nullable column
        "InternalNotes": ["Audit pending", "VIP", "Standard"],  # Unexpected column
    })

    invalid_report = pp.schema_report(invalid_df, schema=contract, allow_unexpected=False)
    print(invalid_report[["Column", "Check", "Expected", "Actual", "Status", "Message"]])

    failed_checks = invalid_report[invalid_report["Status"] == "FAIL"]
    print(f"\nTotal Failures Flagged: {len(failed_checks)}")
    for _, row in failed_checks.iterrows():
        print(f"  [FAIL] {row['Column']} ({row['Check']}): {row['Message']}")

    print("\nSchema validation completed successfully!")


if __name__ == "__main__":
    main()
