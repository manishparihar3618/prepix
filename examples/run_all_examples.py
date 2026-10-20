"""
Automated Examples Verification Runner
Executes all example scripts in the examples/ directory to ensure 100% runnability.
"""
import glob
import os
import subprocess
import sys
import time


def main():
    print("=" * 70)
    print(" PREPIX AUTOMATED EXAMPLES RUNNER")
    print("=" * 70)

    examples_dir = os.path.dirname(os.path.abspath(__file__))
    example_files = sorted(glob.glob(os.path.join(examples_dir, "*.py")))
    # Exclude the runner itself
    example_files = [f for f in example_files if os.path.basename(f) != "run_all_examples.py"]

    python_exe = sys.executable
    print(f"Using Python interpreter: {python_exe}")
    print(f"Discovered {len(example_files)} example script(s).\n")

    passed = 0
    failed = 0

    for script_path in example_files:
        script_name = os.path.basename(script_path)
        print(f"--> Running: {script_name:<30} ... ", end="", flush=True)

        start_time = time.perf_counter()
        result = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.abspath(os.path.join(examples_dir, "..")),
        )
        duration = time.perf_counter() - start_time

        if result.returncode == 0:
            passed += 1
            print(f"[PASS] ({duration:.2f}s)")
        else:
            failed += 1
            print(f"[FAIL] ({duration:.2f}s)")
            print("\n" + "=" * 60)
            print(f"ERROR OUTPUT FOR: {script_name}")
            print("=" * 60)
            print(result.stdout)
            print(result.stderr)
            print("=" * 60 + "\n")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED out of {len(example_files)} total.")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("All examples executed and verified successfully!")


if __name__ == "__main__":
    main()
