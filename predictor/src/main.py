import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Imports are now relative within the 'evio' package
from src.evio_in.pacer import Pacer
from src.evio_in.dat_file import DatFileSource
from src.evio_in.play_dat import get_frame, get_window
from src.yolo.yolo import detect_drone_crop
from src.roo.aoi_detection import detect_aois
from src.rse.rpm_estimation import estimate_rpm_from_events, RPMConfig


def apply_crop_mask(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, crop
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply crop region to event coordinates."""
    if crop is None:
        return x, y, t

    crop_mask = (x >= crop.x1) & (x < crop.x2) & (y >= crop.y1) & (y < crop.y2)
    return x[crop_mask], y[crop_mask], t[crop_mask]


def trim_event_buffer(
    buf_x: np.ndarray,
    buf_y: np.ndarray,
    buf_t: np.ndarray,
    max_window_us: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only recent events within max_window_us of latest timestamp."""
    if buf_t.size == 0:
        return buf_x, buf_y, buf_t

    t_now = buf_t.max()
    keep_mask = buf_t >= (t_now - max_window_us)
    return buf_x[keep_mask], buf_y[keep_mask], buf_t[keep_mask]


def update_aois_if_needed(
    buf_x: np.ndarray,
    buf_y: np.ndarray,
    buf_t: np.ndarray,
    last_update_us: int | None,
    aoi_window_us: int,
    num_clusters: int,
    update_interval_us: int = 50_000,
) -> tuple[list, int | None]:
    """Update AOI detection periodically."""
    if buf_t.size == 0:
        return [], last_update_us

    t_now = buf_t.max()

    # Check if update is needed
    if last_update_us is not None and (t_now - last_update_us) < update_interval_us:
        return [], last_update_us  # Return empty to signal no update

    # Select events from AOI window
    aoi_start = t_now - aoi_window_us
    aoi_mask = buf_t >= aoi_start
    if not np.any(aoi_mask):
        return [], last_update_us

    # Detect AOIs
    aois = detect_aois(
        buf_x[aoi_mask],
        buf_y[aoi_mask],
        width=1280,
        height=720,
        num_clusters=num_clusters,
    )

    return aois, t_now


def estimate_aoi_rpms(
    buf_x: np.ndarray,
    buf_y: np.ndarray,
    buf_t: np.ndarray,
    aois: list,
    rpm_window_us: int,
    rpm_config: RPMConfig,
) -> dict[int, float]:
    """Estimate RPM for each detected AOI."""
    if buf_t.size == 0 or len(aois) == 0:
        return {}

    t_now = buf_t.max()
    rpm_start = t_now - rpm_window_us
    rpm_mask = buf_t >= rpm_start

    if not np.any(rpm_mask):
        return {}

    xr = buf_x[rpm_mask]
    yr = buf_y[rpm_mask]
    tr = buf_t[rpm_mask]

    rpm_estimates = {}
    for idx, aoi in enumerate(aois):
        x1, y1, x2, y2 = aoi.bbox
        roi_mask = (xr >= x1) & (xr < x2) & (yr >= y1) & (yr < y2)

        if np.any(roi_mask):
            rpm = estimate_rpm_from_events(
                xr[roi_mask], yr[roi_mask], tr[roi_mask], aoi.bbox, rpm_config
            )
            if rpm is not None:
                rpm_estimates[idx] = rpm

    return rpm_estimates


def update_rpm_history(
    rpm_history: dict[int, list[float]],
    new_rpms: dict[int, float],
) -> dict[int, list[float]]:
    """Update RPM history with new measurements."""
    updated = rpm_history.copy()
    for aoi_idx, rpm in new_rpms.items():
        if aoi_idx not in updated:
            updated[aoi_idx] = []
        updated[aoi_idx].append(rpm)
    return updated


def compute_average_rpms(rpm_history: dict[int, list[float]]) -> dict[int, float]:
    """Compute average RPM for each AOI from history."""
    return {
        aoi_idx: float(np.mean(rpms))
        for aoi_idx, rpms in rpm_history.items()
        if len(rpms) > 0
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stream a .dat file with real-time pacing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .dat file path.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time).",
    )
    parser.add_argument(
        "--no-drop",
        action="store_false",
        dest="force_speed",
        help="Disable frame dropping. The pacer will process every batch, even if it falls behind.",
    )
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=0.1,
        help="Lag tolerance in seconds before dropping batches (default: 0.1s).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10000 * 15,  # 150ms as per paper
        help="Batch window size in microseconds (default: 10000 µs = 10ms).",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="Number of rotating objects (AOIs) to detect (default: 1).",
    )
    parser.add_argument(
        "--symmetry",
        type=int,
        default=3,
        help="Blade symmetry order for RPM calculation (default: 3).",
    )
    return parser.parse_args()


def main():
    """Main streaming loop."""
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found at '{args.input}'")
        sys.exit(1)

    print("--- Configuration ---")
    print(f"Input file: {args.input}")
    print(f"Playback speed: {args.speed}x")
    print(f"AOI clusters: {args.num_clusters}")
    print(f"Blade symmetry: {args.symmetry}")
    print("---------------------")

    src = DatFileSource(
        args.input, width=1280, height=720, window_length_us=args.window
    )
    pacer = Pacer(
        speed=args.speed,
        force_speed=args.force_speed,
        drop_tolerance_s=args.drop_tolerance,
    )

    if len(src) == 0:
        print("\nError: No batches were generated from the source file.")
        print(
            "This might be due to a very large --window size for a short recording, or an empty input file."
        )
        sys.exit(1)

    print(f"\n--- Starting Paced Playback ({len(src)} batches) ---")
    start_time = time.perf_counter()

    # RPM estimation configuration
    rpm_config = RPMConfig(symmetry_order=args.symmetry)

    # Buffers for streaming AOI/RPM estimation
    event_buffer_x = np.array([], dtype=np.int64)
    event_buffer_y = np.array([], dtype=np.int64)
    event_buffer_t = np.array([], dtype=np.int64)
    aoi_window_us = 150_000  # 150ms for AOI detection
    rpm_window_us = 30_000  # 30ms for RPM estimation
    aois = []
    last_aoi_update_us = None
    rpm_history: dict[int, list[float]] = {}  # Track RPM history per AOI

    print("\n--- Data Import Complete ---")
    for batch_range in pacer.pace(src.ranges()):
        # Extract event data for this batch
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        x_coords, y_coords = window[0], window[1]
        timestamps = src.timestamps_sorted[batch_range.start : batch_range.stop]

        # Optional: detect and apply drone crop region
        frame = get_frame(window)
        drone_crop = detect_drone_crop(frame)
        x_coords, y_coords, timestamps = apply_crop_mask(
            x_coords, y_coords, timestamps, drone_crop
        )

        # Append to buffer
        event_buffer_x = np.concatenate([event_buffer_x, x_coords])
        event_buffer_y = np.concatenate([event_buffer_y, y_coords])
        event_buffer_t = np.concatenate([event_buffer_t, timestamps])

        # Trim buffer to keep only recent events
        max_window = max(aoi_window_us, rpm_window_us)
        event_buffer_x, event_buffer_y, event_buffer_t = trim_event_buffer(
            event_buffer_x, event_buffer_y, event_buffer_t, max_window
        )

        # Update AOIs periodically (every ~50ms)
        new_aois, new_update_time = update_aois_if_needed(
            event_buffer_x,
            event_buffer_y,
            event_buffer_t,
            last_aoi_update_us,
            aoi_window_us,
            args.num_clusters,
        )
        if len(new_aois) > 0:
            aois = new_aois
            last_aoi_update_us = new_update_time

        # Estimate RPM for each AOI
        rpm_estimates = estimate_aoi_rpms(
            event_buffer_x,
            event_buffer_y,
            event_buffer_t,
            aois,
            rpm_window_us,
            rpm_config,
        )

        # Update RPM history and compute averages
        rpm_history = update_rpm_history(rpm_history, rpm_estimates)
        avg_rpms = compute_average_rpms(rpm_history)

        # Display status
        wall_time = time.perf_counter() - start_time
        rpm_str = ", ".join(
            f"AOI{i}={rpm:.0f}(avg:{avg_rpms.get(i, 0):.0f})"
            for i, rpm in rpm_estimates.items()
        )
        print(
            f"\rWall Time: {wall_time: >6.2f}s | "
            f"Event Time: {batch_range.end_ts_us / 1e6: >6.2f}s | "
            f"Emitted: {pacer.emitted_batches: >5} | "
            f"Dropped: {pacer.dropped_batches: >5} | "
            f"AOIs: {len(aois)} | RPM: [{rpm_str}]   ",
            end="",
        )

    print("\n\n--- Playback Finished ---")

    # Display final average RPMs
    if rpm_history:
        print("\n--- Final Average RPMs ---")
        final_avgs = compute_average_rpms(rpm_history)
        for aoi_idx in sorted(final_avgs.keys()):
            avg_rpm = final_avgs[aoi_idx]
            sample_count = len(rpm_history[aoi_idx])
            print(f"  AOI {aoi_idx}: {avg_rpm:.2f} RPM (n={sample_count} measurements)")
    else:
        print("\nNo RPM measurements recorded.")


if __name__ == "__main__":
    main()
