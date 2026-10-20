# Installation Guide

This guide covers installing Prepix, platform and Python compatibility, dependency requirements, and troubleshooting common installation hurdles.

---

## System Requirements & Compatibility

Prepix is tested and verified across multiple Python environments:

| Component | Supported Range | Tested Version | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | `>= 3.9` | `3.11.15` | Fully compatible with Python 3.9, 3.10, 3.11, 3.12, and 3.13 |
| **pandas** | `>= 2.0.0` | `3.0.6` | Supports nullable dtypes and standard pandas extension types |
| **numpy** | `>= 1.24.0` | `2.4.6` | Supports NumPy 1.x and NumPy 2.x |
| **Operating System** | Cross-platform | Windows, macOS, Linux | Pure Python package with no compiled C binaries |

---

## 1. Installing During Development (Local Repository)

Before official public release to PyPI, you can install Prepix directly from the source repository in editable mode:

```bash
# Clone the repository
git clone https://github.com/manishparihar3618/prepix.git
cd prepix

# Recommended: Create and activate an isolated virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

# Install prepix in editable development mode
pip install --upgrade pip
pip install -e .
```

The `-e` flag (editable mode) symlinks your package source directory `src/prepix` into your Python environment so that any changes to code or documentation examples are reflected immediately without reinstalling.

---

## 2. Installing from PyPI (Post-Release)

Once Prepix is officially published to the Python Package Index (PyPI), users will be able to install it directly with standard `pip`:

```bash
pip install prepix
```

*Note: For production deployments, specify a version pin (e.g., `prepix==0.1.0`) in your `requirements.txt` or `pyproject.toml`.*

---

## 3. Verifying Installation

Verify that Prepix and its core dependencies are imported cleanly:

```bash
python -c "import prepix as pp; import pandas as pd; print(f'Prepix version: {pp.__version__}'); print('Successfully imported Prepix!')"
```

Expected output:
```text
Prepix version: 0.1.0
Successfully imported Prepix!
```

---

## 4. Common Installation Issues and Troubleshooting

### Issue A: `No module named 'prepix'`
- **Cause**: The package was installed in a different Python environment than the one currently active in your terminal or IDE.
- **Solution**: Confirm which Python binary is running (`which python` or `Get-Command python`), ensure your intended `.venv` is activated, and re-run `pip install -e .`.

### Issue B: Older pandas or numpy Version Conflict
- **Cause**: Existing project pins older versions such as `pandas<2.0.0` or `numpy<1.24.0`.
- **Solution**: Prepix requires modern pandas memory semantics and array interfaces. Upgrade dependencies using:
  ```bash
  pip install --upgrade "pandas>=2.0.0" "numpy>=1.24.0"
  ```

### Issue C: `setuptools` build error during `pip install -e .`
- **Cause**: Outdated `pip` or `setuptools` build frontend that does not support PEP 660 editable builds.
- **Solution**: Upgrade pip and setuptools:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```
