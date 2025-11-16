# src/profiling.py
import time
from contextlib import contextmanager
from typing import Dict, List

Timings = Dict[str, List[float]]


def create_timings() -> Timings:
    return {}


def add_timing(timings: Timings, name: str, duration_s: float) -> None:
    """
    Add a timing sample (in seconds) under the given name.
    """
    if name not in timings:
        timings[name] = []
    timings[name].append(duration_s)


@contextmanager
def timing_section(timings: Timings, name: str):
    """
    Context manager for timing a code section.

    Example:
        with timing_section(timings, "extract_roi_intensity"):
            extract_roi_intensity(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        add_timing(timings, name, t1 - t0)


def log_timings(logger, timings: Timings, title: str = "Timing stats") -> None:
    """
    Log all timing statistics using the provided logger.
    """
    if not timings:
        return

    logger.info("\n=== %s ===", title)
    for name, samples in timings.items():
        if not samples:
            continue
        avg = sum(samples) / len(samples)
        mn = min(samples)
        mx = max(samples)
        logger.info(
            f"{name:>20}: "
            f"avg={avg * 1000:.2f} ms, "
            f"min={mn * 1000:.2f} ms, "
            f"max={mx * 1000:.2f} ms, "
            f"n={len(samples)}"
        )
