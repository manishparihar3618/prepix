# Examples Guide

This guide introduces the realistic, self-contained example scripts located in the `examples/` directory. Each example can be run directly using Python and utilizes synthetic data with zero external data dependencies.

---

## Directory Overview

| Script | Title / Focus | Functions Demonstrated |
| :--- | :--- | :--- |
| `examples/quickstart.py` | Basic quickstart walkthrough | `report()`, sub-reports |
| `examples/business_dataset.py` | End-to-end employee & business data | All functions against `data.csv` |
| `examples/missing_data.py` | Missing-value auditing & suggestions | `missing_report()`, threshold tuning |
| `examples/column_profiling.py` | Column metadata & memory estimation | `column_summary()`, suggested types |
| `examples/duplicate_analysis.py` | Row & key-level duplicate detection | `duplicate_report()`, `subset`, `keep` |
| `examples/unique_cardinality.py` | Cardinality tiers & ID detection | `unique_report()`, constant columns |
| `examples/numerical_analysis.py` | Descriptive stats & outlier analysis | `numeric_summary()`, `extended`, percentiles |
| `examples/schema_validation.py` | Schema enforcement & diagnostic logs | `schema_report()`, pass/fail checks |
| `examples/unified_report.py` | Comprehensive unified report & JSON export | `report()`, `.to_dict()`, `.to_json()` |

---

## Running the Examples

You can run each example independently from the root of the repository:

```bash
# Run any single example
python examples/quickstart.py
python examples/schema_validation.py

# Or run the automated validation runner to verify all examples in sequence:
python examples/run_all_examples.py
```

All examples are tested against the package and exit with status code `0`.
