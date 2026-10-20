"""
Prepix Example: Column Profiling
Demonstrates column_summary() to inspect data types, unique counts, memory usage,
and suggested types across diverse data types.
"""
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: COLUMN PROFILING")
    print("=" * 70)

    # Dataset with a wide variety of column types
    df = pd.DataFrame({
        "IsEnrolled": [True, False, True, True, False],
        "Score": [98.5, 87.0, 92.5, 78.0, 88.0],
        "Count": [1, 2, 3, 4, 5],
        "Category": ["Gold", "Silver", "Gold", "Bronze", "Silver"],
        "BinaryCat": ["Yes", "No", "Yes", "Yes", "No"],
        "RegistrationDate": pd.to_datetime(["2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-15"]),
        "Notes": ["Active student", "On leave", "Dean's list", "Audit only", "Active student"],
    })

    print("Input Dataset Preview:")
    print(df)

    # Generate column summary
    print("\n" + "=" * 70)
    print(" COLUMN SUMMARY (pp.column_summary(df)):")
    print("=" * 70)
    summary = pp.column_summary(df)
    print(summary)

    print("\nSummary Interpretation:")
    print("- Data Type: Native pandas dtype for each column.")
    print("- Memory(Bytes): Deep memory footprint including Python string object heaps.")
    print("- Suggested Type: Semantic classification heuristic:")
    print("    * 'Binary': Boolean columns.")
    print("    * 'Numeric': Float and integer features.")
    print("    * 'Categorical (2 values)': Non-numeric columns with exactly 2 categories.")
    print("    * 'Categorical': Multi-category text or object features.")
    print("    * 'Datetime': Timestamp and datetime columns.")

    print("\nColumn profiling completed successfully!")


if __name__ == "__main__":
    main()
