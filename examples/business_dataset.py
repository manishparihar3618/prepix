"""
Prepix Example: Business Dataset Quality Audit
Demonstrates an end-to-end quality audit on a multi-department organizational dataset.
"""
import os
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: BUSINESS DATASET AUDIT")
    print("=" * 70)

    # Load synthetic dataset data.csv or generate equivalent synthetic DataFrame
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame({
            "ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 110],
            "Name": ["Aman", "Priya", "Rahul", "Sneha", "Vikram", "Neha", "Amit", "Pooja", "Rohan", "Ananya", "Ananya"],
            "Age": [28.0, 32.0, None, 26.0, 45.0, 29.0, 35.0, None, 31.0, 27.0, 27.0],
            "Department": ["Engineering", "HR", "Sales", "Engineering", "Finance", "Marketing", "Engineering", "HR", "Sales", "Engineering", "Engineering"],
            "Salary": [75000.0, 62000.0, 54000.0, 82000.0, 95000.0, 58000.0, 88000.0, 60000.0, 52000.0, 78000.0, 78000.0],
            "Company_Status": ["Active"] * 11,
        })

    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDataset Preview:")
    print(df.head(5))

    # 1. Structural column audit
    print("\n1. Column Structure & Memory Breakdown:")
    col_summary = pp.column_summary(df)
    print(col_summary[["Column", "Data Type", "Missing Values", "Unique Values", "Suggested Type"]])

    # 2. Missing-value analysis
    print("\n2. Missing-Value Prioritization:")
    missing = pp.missing_report(df, only_missing=True)
    print(missing)

    # 3. Duplicate row check
    print("\n3. Duplicate Record Audit:")
    dup_report = pp.duplicate_report(df)
    print(dup_report)

    # 4. Numeric distribution
    print("\n4. Numeric Distribution Summary:")
    num_summary = pp.numeric_summary(df, extended=True)
    print(num_summary[["Column", "Mean", "Median", "Std", "Min", "Max", "Outliers"]])

    # 5. Schema verification
    print("\n5. Organizational Contract Verification:")
    contract_schema = {
        "ID": {"type": "integer", "nullable": False, "unique": True},
        "Name": "string",
        "Salary": {"type": "numeric", "nullable": False},
        "Department": "string",
    }
    schema_results = pp.schema_report(df, schema=contract_schema, allow_unexpected=True)
    print(schema_results[["Column", "Check", "Expected", "Actual", "Status", "Message"]])

    print("\nBusiness dataset audit completed successfully!")


if __name__ == "__main__":
    main()
