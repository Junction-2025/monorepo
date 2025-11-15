"""
Simplified RPM estimation using correlation-based periodicity detection.

Functional approach based on EE3P methodology:
- Aggregate events into time-sliced frames
- Compute normalized cross-correlation
- Detect peaks to find periodicity
- Convert to RPM accounting for symmetry
"""

import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray


class RPMConfig(NamedTuple):
    """Configuration for RPM estimation."""

    slice_us: int = 500  # Time slice duration in microseconds
    min_slices: int = 10  # Minimum slices needed for correlation
    symmetry_order: int = 3  # Blade/pattern repeats per revolution
    peak_prominence: float = 0.1  # Peak detection threshold
    max_peaks: int = 10  # Maximum peaks to consider


def aggregate_events_to_frames(
    x: NDArray[np.int64],
    y: NDArray[np.int64],
    t: NDArray[np.int64],
    bbox: tuple[int, int, int, int],
    slice_us: int = 500,
) -> tuple[NDArray[np.uint16], NDArray[np.float64]]:
    """
    Aggregate ROI events into time-sliced frames.

    Args:
        x, y, t: Event coordinates and timestamps (microseconds)
        bbox: (x1, y1, x2, y2) ROI bounding box
        slice_us: Duration of each time slice in microseconds

    Returns:
        frames: [S, H, W] event count per pixel per slice
        times: [S] center timestamp of each slice (seconds)
    """
    if len(t) == 0:
        return np.zeros((0, 1, 1), dtype=np.uint16), np.zeros(0, dtype=np.float64)

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    t_min, t_max = t.min(), t.max()
    total_us = t_max - t_min
    num_slices = int(np.ceil(total_us / slice_us))

    if num_slices == 0:
        return np.zeros((0, 1, 1), dtype=np.uint16), np.zeros(0, dtype=np.float64)

    frames = np.zeros((num_slices, h, w), dtype=np.uint16)
    times = np.zeros(num_slices, dtype=np.float64)

    # Assign events to slices
    slice_idx = ((t - t_min) // slice_us).astype(int)
    slice_idx = np.clip(slice_idx, 0, num_slices - 1)

    # Convert to local ROI coordinates
    xr = x - x1
    yr = y - y1

    # Accumulate events into frames
    for xi, yi, si in zip(xr, yr, slice_idx):
        if 0 <= xi < w and 0 <= yi < h:
            frames[si, yi, xi] += 1

    # Compute slice center times
    for s in range(num_slices):
        t_start = t_min + s * slice_us
        t_end = t_start + slice_us
        times[s] = 0.5 * (t_start + t_end) * 1e-6  # Convert to seconds

    return frames, times


def normalized_correlation(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    """
    Compute normalized cross-correlation between two images.

    Returns correlation coefficient in [-1, 1].
    """
    am, bm = a.mean(), b.mean()
    a_centered = a - am
    b_centered = b - bm
    a_std = a.std() + 1e-6
    b_std = b.std() + 1e-6
    num = (a_centered * b_centered).sum()
    den = a_std * b_std * a.size
    return float(num / den)


def find_peaks(signal: NDArray[np.float32], prominence: float = 0.1) -> NDArray[np.int64]:
    """
    Find local maxima in signal above threshold.

    Args:
        signal: 1D array
        prominence: Threshold as multiple of std above mean

    Returns:
        Peak indices
    """
    if signal.size < 3:
        return np.array([], dtype=np.int64)

    mean, std = signal.mean(), signal.std() + 1e-6
    thresh = mean + prominence * std
    peaks = []

    for i in range(1, signal.size - 1):
        if (
            signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
            and signal[i] > thresh
        ):
            peaks.append(i)

    return np.array(peaks, dtype=np.int64)


def estimate_rpm(
    frames: NDArray[np.uint16],
    times: NDArray[np.float64],
    config: RPMConfig = RPMConfig(),
) -> float | None:
    """
    Estimate RPM from aggregated event frames using correlation.

    Process:
    1. Compute correlation of each frame with reference (first) frame
    2. Find peaks in correlation signal
    3. Estimate period from peak spacing
    4. Convert to RPM accounting for symmetry

    Args:
        frames: [S, H, W] event count frames
        times: [S] timestamps in seconds
        config: RPM estimation configuration

    Returns:
        RPM estimate or None if insufficient structure
    """
    num_slices = frames.shape[0]
    if num_slices < config.min_slices:
        return None

    # Use first frame as reference
    ref = frames[0].astype(np.float32)

    # Compute correlation with each frame
    corr = np.zeros(num_slices, dtype=np.float32)
    for s in range(num_slices):
        frame_s = frames[s].astype(np.float32)
        corr[s] = normalized_correlation(ref, frame_s)

    # Find peaks in correlation signal
    peaks = find_peaks(corr, config.peak_prominence)
    if peaks.size < 2:
        return None

    # Limit number of peaks for robustness
    if peaks.size > config.max_peaks:
        peaks = peaks[: config.max_peaks]

    # Compute intervals between peaks
    peak_times = times[peaks]
    intervals = np.diff(peak_times)
    intervals = intervals[intervals > 0]

    if intervals.size == 0:
        return None

    # Median period between pattern repetitions
    median_period = np.median(intervals)
    if median_period <= 0:
        return None

    # Convert to RPM
    f_pattern = 1.0 / median_period  # Hz of pattern repetition
    f_rotation = f_pattern / config.symmetry_order  # Hz of full rotation
    rpm = 60.0 * f_rotation

    return float(rpm)


def estimate_rpm_from_events(
    x: NDArray[np.int64],
    y: NDArray[np.int64],
    t: NDArray[np.int64],
    bbox: tuple[int, int, int, int],
    config: RPMConfig = RPMConfig(),
) -> float | None:
    """
    Convenience function: estimate RPM directly from ROI events.

    Args:
        x, y, t: Event coordinates and timestamps
        bbox: ROI bounding box
        config: RPM estimation configuration

    Returns:
        RPM estimate or None
    """
    frames, times = aggregate_events_to_frames(x, y, t, bbox, config.slice_us)

    if frames.shape[0] < config.min_slices:
        return None

    return estimate_rpm(frames, times, config)
