"""
Prepix Example: Unified Quality Report & Export
Demonstrates pp.report(df), inspecting the resulting PrepixReport container,
and exporting results to Python dict and JSON.
"""
import json
import pandas as pd
import prepix as pp


def main():
    print("=" * 70)
    print(" PREPIX EXAMPLE: UNIFIED REPORT & SERIALIZATION")
    print("=" * 70)

    # Realistic retail store dataset
    df = pd.DataFrame({
        "StoreCode": ["ST101", "ST102", "ST103", "ST104", "ST104"],
        "City": ["Seattle", "Denver", "Austin", "Boston", "Boston"],
        "QuarterlySales": [245000.0, 180000.0, None, 310000.0, 310000.0],
        "FootTraffic": [12000, 9500, 14200, 18500, 18500],
        "OperationalStatus": ["Open"] * 5,  # Zero-variance constant
    })

    # Generate unified quality report
    report = pp.report(df)

    # 1. Print formatted console report
    print("\n1. Formatted CLI Report Output:")
    print(report)

    # 2. Inspect Issues Table
    print("\n2. Prioritized Issues Table (report.issues):")
    print(report.issues)

    # 3. Export to Python dictionary
    print("\n3. Serializing Report to Dictionary:")
    report_dict = report.to_dict()
    print(f"Dictionary keys: {list(report_dict.keys())}")
    print(f"Total issue records exported: {len(report_dict['issues'])}")

    # 4. Export to JSON string
    print("\n4. Serializing Report to JSON String:")
    json_str = report.to_json(indent=2)
    parsed = json.loads(json_str)
    print(f"Valid JSON generated! Root keys count: {len(parsed)}")

    print("\nUnified report and serialization completed successfully!")


if __name__ == "__main__":
    main()
