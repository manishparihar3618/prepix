# Contributing to Prepix

Thank you for your interest in contributing to Prepix! We welcome contributions that help improve data quality inspection, documentation, test coverage, and developer experience.

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive, and collaborative in all discussions and code reviews.

---

## Development Setup

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- `git`
- Virtual environment tool (such as Python's built-in `venv`)

### 2. Clone the Repository
```bash
git clone https://github.com/manishparihar3618/prepix.git
cd prepix
```

### 3. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install in Editable Mode
```bash
pip install --upgrade pip
pip install -e .
```

---

## Core Development Guidelines & Architecture

Prepix adheres strictly to the following engineering principles:

1. **DataFrame Immutability**:
   - Analysis functions MUST NOT mutate the caller's DataFrame in place. Never perform in-place column assignments, drops, or type casts on the input data.
2. **Transparent Recommendations**:
   - Suggestions produced in reports (such as imputation methods or dropping constant columns) are advisory heuristics. Prepix does not silently modify datasets.
3. **Resilient Edge-Case Handling**:
   - Empty DataFrames (`0x0`, `0xN`, `Nx0`).
   - Duplicate column headers (e.g. `df[["A", "A"]]`).
   - Infinite values (`np.inf`, `-np.inf`).
   - Unhashable objects (lists, dictionaries) in cell values.
4. **Minimal External Dependencies**:
   - Runtime dependencies are strictly limited to `pandas>=2.0.0` and `numpy>=1.24.0`. Do not introduce heavy extra dependencies for core features.

---

## Running the Test Suite

All tests use Python's built-in `unittest` runner, requiring no external test runner to be installed:

```bash
# Run the entire test suite
python -m unittest discover -s tests -p "test_*.py" -v

# Run a specific test module
python -m unittest tests/test_schema_report.py

# Run a specific test case or method
python -m unittest tests.test_quality.TestQualityFunctions.test_missing_report_dataset_a
```

Ensure all tests pass before submitting any pull request.

---

## Running Benchmarks and Examples

Verify that examples and benchmarks execute without warnings:

```bash
# Execute performance benchmarks
python benchmark.py

# Execute all examples
python examples/run_all_examples.py
```

---

## How to Submit a Contribution

1. **Check Issues First**:
   - Before implementing large changes, search existing [GitHub Issues](https://github.com/manishparihar3618/prepix/issues) or open a discussion to align on design and scope.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-feature-name
   ```
3. **Write Unit Tests**:
   - Every bug fix or feature improvement must include corresponding tests in `tests/`.
4. **Commit Your Changes**:
   - Use clear, descriptive commit messages (e.g., `docs: update numeric summary parameter reference`).
5. **Open a Pull Request**:
   - Provide a clear summary of what changed, why the change was made, and link relevant issues.
