"""
Prepix Example: Numerical Feature Analysis
Demonstrates numeric_summary() with standard descriptive statistics,
extended distribution diagnostics (IQR, skewness, kurtosis, outliers), and custom percentiles.
"""
import numpy as np
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: NUMERICAL ANALYSIS")
    print("=" * 70)

    # Generate synthetic numerical data with deliberate characteristics
    np.random.seed(42)
    n = 200

    # 1. Normal distribution (symmetric, few outliers)
    normal_data = np.random.normal(loc=100.0, scale=15.0, size=n)

    # 2. Right-skewed distribution (e.g. household income with extreme high earners)
    skewed_data = np.random.exponential(scale=50000.0, size=n)
    skewed_data[0] = 750000.0  # Deliberate extreme high outlier

    # 3. Data with negatives and zeros (e.g. net profit/loss)
    profit_data = np.random.normal(loc=50.0, scale=100.0, size=n)
    profit_data[1:10] = 0.0

    df = pd.DataFrame({
        "NormalFeature": normal_data,
        "SkewedIncome": skewed_data,
        "NetProfit": profit_data,
        "NonNumericCol": ["Text"] * n,  # Will be cleanly excluded automatically
    })

    print(f"Dataset: {df.shape[0]} rows with 3 numerical columns and 1 text column.")

    # 1. Standard 5-number + summary statistics
    print("\n1. Standard Numeric Summary (Base 12 statistics):")
    base_summary = pp.numeric_summary(df)
    print(base_summary[["Column", "Count", "Mean", "Median", "Std", "Min", "Max", "Range"]])

    # 2. Extended diagnostics (IQR, Skewness, Kurtosis, Zeros, Negatives, Outliers)
    print("\n2. Extended Distribution Diagnostics (extended=True):")
    ext_summary = pp.numeric_summary(df, extended=True)
    print(ext_summary[["Column", "IQR", "Skewness", "Kurtosis", "Zeros", "Negatives", "Outliers"]])

    # 3. Custom percentiles
    print("\n3. Custom Percentiles (percentiles=[0.01, 0.05, 0.95, 0.99]):")
    pct_summary = pp.numeric_summary(df, percentiles=[0.01, 0.05, 0.95, 0.99])
    print(pct_summary[["Column", "Min", "P1", "P5", "Median", "P95", "P99", "Max"]])

    print("\nDiagnostic Notes:")
    print("- Positive Skewness (> 1.0) indicates a long right tail (e.g. high-income outliers).")
    print("- Outliers count observations beyond the Tukey fence [Q1 - 1.5*IQR, Q3 + 1.5*IQR].")
    print("- Non-numeric columns (like 'NonNumericCol') are automatically filtered out.")

    print("\nNumerical analysis completed successfully!")


if __name__ == "__main__":
    main()
