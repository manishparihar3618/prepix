"""
Prepix Example: Unique Values and Cardinality
Demonstrates unique_report() across constant, binary, categorical, and identifier columns,
and clarifies the dropna parameter behavior.
"""
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: UNIQUENESS & CARDINALITY")
    print("=" * 70)

    # Dataset with distinct cardinality levels
    df = pd.DataFrame({
        "SystemUUID": [f"user_{i:04d}" for i in range(10)],                  # 100% Unique
        "CountryCode": ["US", "US", "US", "CA", "US", "GB", "US", "FR", "US", "DE"],  # Medium diversity
        "IsSubscribed": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "Yes"],  # Binary
        "Environment": ["Production"] * 10,                                  # Constant (Zero Variance)
        "Department": ["Sales", "Engineering", "HR", "Sales", "Sales", "HR", "Sales", "Sales", "Sales", "Sales"],
        "OptionalTag": ["alpha", "beta", "alpha", None, None, "alpha", "gamma", "alpha", None, "beta"],
    })

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1. Standard Uniqueness Report (dropna=False default)
    print("\n1. Standard Cardinality Report (pp.unique_report(df, dropna=False)):")
    rep_standard = pp.unique_report(df)
    print(rep_standard[["Column", "Unique Values", "Unique Percentage(%)", "Cardinality", "Status"]])

    # 2. Compare dropna=False vs dropna=True on columns with nulls
    print("\n2. Comparing dropna=False vs dropna=True on 'OptionalTag':")
    rep_with_nulls = pp.unique_report(df, dropna=False)
    rep_no_nulls = pp.unique_report(df, dropna=True)

    tag_with = rep_with_nulls[rep_with_nulls["Column"] == "OptionalTag"]["Unique Values"].iloc[0]
    tag_without = rep_no_nulls[rep_no_nulls["Column"] == "OptionalTag"]["Unique Values"].iloc[0]

    print(f"   dropna=False (default): {tag_with} unique values (counts None/NaN as a distinct category)")
    print(f"   dropna=True:           {tag_without} unique values (ignores None/NaN, matching pd.Series.nunique())")

    print("\nCardinality Tiers Explained:")
    print("- 'Constant': Exactly 1 unique value across all rows (zero predictive variance).")
    print("- 'Very High (Unique)': 100% unique values (candidate primary key / identifier).")
    print("- 'Very High': >= 90% unique values.")
    print("- 'High': 50% - 90% unique values.")
    print("- 'Medium': 10% - 50% unique values.")
    print("- 'Low': < 10% unique values.")

    print("\nUniqueness and cardinality audit completed successfully!")


if __name__ == "__main__":
    main()
