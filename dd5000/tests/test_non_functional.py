"""
Non-functional test battery for RPM detection.

Tests the predictor against known scenarios with expected RPM ranges.
"""

import subprocess
import glob
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TestCase:
    """Defines a test scenario with expected RPM range."""

    name: str
    filename: str
    expected_min: float
    expected_max: float
    notes: str = ""


# Test scenarios
TEST_CASES = [
    TestCase(
        name="Drone Idle",
        filename="drone_idle.dat",
        expected_min=5000,
        expected_max=6000,
        notes="Stationary drone at ~100m, yolo-verified frames only",
    ),
    TestCase(
        name="Drone Moving",
        filename="drone_moving.dat",
        expected_min=5500,
        expected_max=6500,
        notes="Moving drone at ~100m, ~20s duration",
    ),
]


def run_predictor(file_path: Path) -> float | None:
    """
    Run the predictor on a file and extract the average RPM.

    Returns:
        Average RPM if successful, None if failed or no measurements
    """
    # Use the project's .venv Python
    project_dir = Path(__file__).parent.parent
    venv_python = project_dir / ".venv" / "bin" / "python"

    # Get the log directory and note existing log files before running
    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    existing_logs = set(log_dir.glob("*.txt"))

    cmd = [
        str(venv_python),
        "-m",
        "src.main",
        "--input",
        str(file_path),
        "--no-display",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        # Parse output for "RPM: Y.YY" from logger
        # Check both stdout and stderr as logger might output to either
        combined_output = result.stdout + "\n" + result.stderr

        for line in combined_output.splitlines():
            # Look for the final average RPM output
            # Format: "INFO - RPM: 1234.56" (without blade_count in the line)
            if "RPM:" in line and "blade_count" not in line:
                # Extract RPM value from "INFO - RPM: 1234.56"
                parts = line.split("RPM:")
                if len(parts) >= 2:
                    rpm_str = parts[1].strip()
                    try:
                        return float(rpm_str)
                    except ValueError:
                        continue

        return None

    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 120s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def calculate_error(value: float, min_val: float, max_val: float) -> float:
    """
    Calculate how far the value is from the expected range.

    Returns:
        0 if within range, otherwise the distance to nearest boundary
    """
    if min_val <= value <= max_val:
        return 0.0
    elif value < min_val:
        return min_val - value
    else:
        return value - max_val


def run_tests():
    """Run all test cases and report results."""
    data_dir = Path(__file__).parent.parent.parent / "data"

    print("=" * 70)
    print("RPM Detection Test Battery")
    print("=" * 70)
    print(f"\nData directory: {data_dir}\n")

    results = []

    for test_case in TEST_CASES:
        file_path = data_dir / test_case.filename

        print(f"Test: {test_case.name}")
        print(f"  File: {test_case.filename}")
        print(
            f"  Expected range: [{test_case.expected_min:.2f}, {test_case.expected_max:.2f}] RPM"
        )

        if not file_path.exists():
            print("  SKIPPED - File not found\n")
            results.append((test_case.name, None, None, "SKIPPED"))
            continue

        print("  Running predictor...")
        measured_rpm = run_predictor(file_path)

        if measured_rpm is None:
            print("  FAILED - No RPM measurement obtained\n")
            results.append((test_case.name, None, None, "FAILED"))
            continue

        error = calculate_error(
            measured_rpm, test_case.expected_min, test_case.expected_max
        )

        print(f"  Measured: {measured_rpm:.2f} RPM")

        if error == 0:
            print("  PASS - Within expected range")
            status = "PASS"
        else:
            print(f"  FAIL - Off by {error:.2f} RPM")
            status = "FAIL"

        print()
        results.append((test_case.name, measured_rpm, error, status))

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    for name, rpm, error, status in results:
        if status == "SKIPPED":
            print(f"{name:30s} - SKIPPED")
        elif status == "FAILED":
            print(f"{name:30s} - FAILED (no measurement)")
        elif status == "PASS":
            print(f"{name:30s} - PASS ({rpm:.2f} RPM)")
        else:
            print(f"{name:30s} - FAIL ({rpm:.2f} RPM, off by {error:.2f})")

    print()


if __name__ == "__main__":
    run_tests()
