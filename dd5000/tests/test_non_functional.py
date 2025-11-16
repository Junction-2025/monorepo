"""
Non-functional test battery for RPM detection.

Tests the predictor against known scenarios with expected RPM ranges.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestCase:
    """Defines a test scenario with expected RPM range."""

    name: str
    filename: str
    expected_min: float
    expected_max: float
    num_clusters: int = 1
    symmetry: int = 3
    notes: str = ""


# Test scenarios
TEST_CASES = [
    TestCase(
        name="Fan Constant RPM",
        filename="fan_const_rpm.dat",
        expected_min=1000,
        expected_max=1200,
        notes="Fan rotating at constant speed, ~10s duration",
    ),
    TestCase(
        name="Fan Varying RPM",
        filename="fan_varying_rpm.dat",
        expected_min=1100,
        expected_max=1300,
        notes="Fan with changing speed, ~20s duration",
    ),
    TestCase(
        name="Fan Varying RPM Turning",
        filename="fan_varying_rpm_turning.dat",
        expected_min=1100,
        expected_max=1300,
        notes="Fan with changing speed and orientation, ~25s duration",
    ),
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


def run_predictor(file_path: Path, num_clusters: int, symmetry: int) -> Optional[float]:
    """
    Run the predictor on a file and extract the average RPM.

    Returns:
        Average RPM if successful, None if failed or no measurements
    """
    # Use the project's .venv Python
    project_dir = Path(__file__).parent.parent
    venv_python = project_dir / ".venv" / "bin" / "python"

    cmd = [
        str(venv_python),
        "-m",
        "src.main",
        "--input",
        str(file_path),
        "--no-display",
        "--no-logging",
        "--speed",
        "5",  # Run at 5x speed for faster testing
        "--num-clusters",
        str(num_clusters),
        "--symmetry",
        str(symmetry),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        # Parse output for "AOI X: Y.YY RPM"
        for line in result.stdout.splitlines():
            if "AOI 0:" in line and "RPM" in line:
                # Extract RPM value from "AOI 0: 1234.56 RPM (n=123)"
                parts = line.split(":")
                if len(parts) >= 2:
                    rpm_part = parts[1].split("RPM")[0].strip()
                    return float(rpm_part)

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
        measured_rpm = run_predictor(
            file_path, test_case.num_clusters, test_case.symmetry
        )

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
