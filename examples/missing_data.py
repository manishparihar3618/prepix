"""
Prepix Example: Missing Data Analysis
Demonstrates missing_report() with configurable thresholds, filtering clean columns,
and interpreting heuristic imputation suggestions.
"""
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: MISSING DATA AUDIT")
    print("=" * 70)

    # Synthetic dataset with diverse missingness patterns across types
    df = pd.DataFrame({
        "PatientID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "BloodPressure": [120, None, 130, 125, None, 140, 115, 128, 132, 122],  # 20% missing
        "HeartRate": [72, 75, 80, 68, 74, None, 70, 76, 71, 73],                 # 10% missing
        "SmokerStatus": ["No", "Yes", None, "No", None, None, "Yes", "No", "No", "Yes"],  # 30% missing
        "GeneticMarker": [None] * 8 + ["TypeA", "TypeB"],                        # 80% missing (severe)
        "HospitalWing": ["WingA"] * 10,                                          # 0% missing (clean)
    })

    print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1. Standard missing values report
    print("\n1. Standard Missing Values Report (Sorted descending by missing %):")
    rep_standard = pp.missing_report(df)
    print(rep_standard)

    # 2. Filter to show only columns with missing data
    print("\n2. Filtered Report (only_missing=True):")
    rep_filtered = pp.missing_report(df, only_missing=True)
    print(rep_filtered)

    # 3. Custom severity thresholds
    print("\n3. Custom Severity Thresholds (Low <= 15%, Medium <= 25%):")
    rep_custom = pp.missing_report(
        df,
        low_threshold=15.0,
        medium_threshold=25.0,
        only_missing=True,
    )
    print(rep_custom)

    print("\nInterpretation Notes:")
    print("- 'Consider Median Imputation': Suggested for numeric columns with <= medium missingness.")
    print("- 'Consider Mode Imputation': Suggested for categorical columns with <= medium missingness.")
    print("- 'Review Before Imputation': Flagged for high-missingness columns (> medium threshold).")

    print("\nMissing data audit completed successfully!")


if __name__ == "__main__":
    main()
