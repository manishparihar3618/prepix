"""
Prepix Example 1: Quickstart Tutorial
Demonstrates loading a realistic dataset and running Prepix's core quality reports.
"""
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: QUICKSTART")
    print("=" * 70)

    # 1. Create a representative employee dataset
    df = pd.DataFrame({
        "EmpID": [101, 102, 103, 104, 105, 105],  # Row 105 duplicated
        "Name": ["Alice", "Bob", "Charlie", "David", "Elena", "Elena"],
        "Department": ["Engineering", "Marketing", "Engineering", "HR", "Sales", "Sales"],
        "Age": [29.0, 34.0, None, 45.0, 31.0, 31.0],  # 1 missing value
        "Salary": [82000.0, 68000.0, 95000.0, 61000.0, 74000.0, 74000.0],
        "Company": ["Acme Corp", "Acme Corp", "Acme Corp", "Acme Corp", "Acme Corp", "Acme Corp"],
    })

    print("\n[Input Dataset Sample (6 rows, 6 columns)]:")
    print(df)

    # 2. Run the Unified Health Report
    print("\n" + "=" * 70)
    print(" 1. GENERATING UNIFIED QUALITY REPORT (pp.report)")
    print("=" * 70)
    report = pp.report(df)
    print(report)

    # 3. Access Granular Sub-Reports
    print("\n" + "=" * 70)
    print(" 2. ACCESSING SPECIFIC SUB-REPORTS")
    print("=" * 70)
    print("\n[Dataset Overview]:")
    print(report.overview)

    print("\n[Key Issues & Findings]:")
    print(report.issues)

    print("\n[Missing Values Summary]:")
    print(report.missing)

    print("\n[Uniqueness & Cardinality]:")
    print(report.unique)

    print("\nQuickstart executed successfully!")


if __name__ == "__main__":
    main()
