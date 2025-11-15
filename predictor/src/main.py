import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Imports are now relative within the 'evio' package
from src.evio_in.pacer import Pacer
from src.evio_in.dat_file import DatFileSource
from src.evio_in.play_dat import get_frame, get_window
from src.yolo.yolo import detect_drone_crop
from src.roo.aoi_detection import detect_aois, AOI
from src.rse.rpm_estimation import estimate_rpm_from_events
from src.roo.rotating_object_extraction import find_heatmap
from src.roo.kmeans import locate_centroids

# Configuration and logging
from src.config import (
    RPMConfig,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    AOI_WINDOW_US,
    AOI_UPDATE_INTERVAL_US,
    RPM_WINDOW_US,
    BATCH_WINDOW_US,
    EVENT_BUFFER_MAX_US,
    BBOX_COLOR,
    TEXT_COLOR,
    CENTROID_COLOR,
    TEXT_FONT,
    TEXT_SCALE,
    TEXT_THICKNESS,
    ENABLE_FRAME_LOGGING,
    ENABLE_HEATMAP_LOGGING,
    ENABLE_LABEL_LOGGING,
    LOG_EVERY_N_FRAMES,
)
from src.logging_utils import (
    setup_run_context,
    save_frame,
    save_heatmap,
    save_labels,
    log_aoi_detection,
    log_rpm_estimate,
    finalize_run,
)


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
    num_clusters: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> tuple[list, int | None, np.ndarray | None, np.ndarray | None]:
    """
    Update AOI detection periodically.

    Returns:
        (aois, update_time, heatmap, labels) - heatmap/labels for logging
    """
    if buf_t.size == 0:
        return [], last_update_us, None, None

    t_now = buf_t.max()

    # Check if update is needed
    if last_update_us is not None and (t_now - last_update_us) < AOI_UPDATE_INTERVAL_US:
        return [], last_update_us, None, None

    # Select events from AOI window
    aoi_start = t_now - AOI_WINDOW_US
    aoi_mask = buf_t >= aoi_start
    if not np.any(aoi_mask):
        return [], last_update_us, None, None

    # Generate heatmap for logging
    heatmap = find_heatmap(
        buf_x[aoi_mask], buf_y[aoi_mask], height=height, width=width
    )

    # Generate labels for logging
    labels, _, _ = locate_centroids(heatmap, k=num_clusters)

    # Detect AOIs
    aois = detect_aois(
        buf_x[aoi_mask],
        buf_y[aoi_mask],
        width=width,
        height=height,
        num_clusters=num_clusters,
    )

    return aois, t_now, heatmap, labels


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


def draw_aoi_bboxes(
    frame: np.ndarray,
    aois: list[AOI],
    rpm_estimates: dict[int, float],
    avg_rpms: dict[int, float],
) -> None:
    """
    Draw bounding boxes and RPM labels on frame (uses config constants).

    Args:
        frame: Image to draw on (modified in-place)
        aois: List of detected AOIs
        rpm_estimates: Current RPM per AOI
        avg_rpms: Average RPM per AOI
    """
    for idx, aoi in enumerate(aois):
        x1, y1, x2, y2 = aoi.bbox

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        # Prepare RPM text
        current_rpm = rpm_estimates.get(idx)
        avg_rpm = avg_rpms.get(idx)

        if current_rpm is not None:
            if avg_rpm is not None:
                rpm_text = f"AOI{idx}: {current_rpm:.0f} (avg:{avg_rpm:.0f})"
            else:
                rpm_text = f"AOI{idx}: {current_rpm:.0f}"
        else:
            rpm_text = f"AOI{idx}: --"

        # Draw RPM text above bounding box
        text_y = max(y1 - 10, 20)
        cv2.putText(
            frame,
            rpm_text,
            (x1, text_y),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        # Draw centroid
        cx, cy = aoi.centroid
        cv2.circle(frame, (int(cx), int(cy)), 3, CENTROID_COLOR, -1)


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
        default=BATCH_WINDOW_US,
        help=f"Batch window size in microseconds (default: {BATCH_WINDOW_US}µs).",
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
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable visualization window (default: False).",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable file logging (frames/heatmaps/logs).",
    )
    return parser.parse_args()


def main():
    """Main streaming loop with logging."""
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found at '{args.input}'")
        sys.exit(1)

    # Setup logging context
    log_ctx = None if args.no_logging else setup_run_context()

    print("--- Configuration ---")
    print(f"Input file: {args.input}")
    print(f"Playback speed: {args.speed}x")
    print(f"AOI clusters: {args.num_clusters}")
    print(f"Blade symmetry: {args.symmetry}")
    print(f"Display: {'disabled' if args.no_display else 'enabled'}")
    print(f"Logging: {'disabled' if args.no_logging else f'enabled -> {log_ctx.run_dir}'}")
    print("---------------------")

    if log_ctx:
        log_ctx.logger.info(f"Input file: {args.input}")
        log_ctx.logger.info(f"Speed: {args.speed}x, Clusters: {args.num_clusters}, Symmetry: {args.symmetry}")

    src = DatFileSource(
        args.input, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, window_length_us=args.window
    )
    pacer = Pacer(
        speed=args.speed,
        force_speed=args.force_speed,
        drop_tolerance_s=args.drop_tolerance,
    )

    if len(src) == 0:
        print("\nError: No batches were generated from the source file.")
        sys.exit(1)

    print(f"\n--- Starting Paced Playback ({len(src)} batches) ---")
    start_time = time.perf_counter()

    # RPM estimation configuration
    rpm_config = RPMConfig(symmetry_order=args.symmetry)

    # Buffers for streaming AOI/RPM estimation
    event_buffer_x = np.array([], dtype=np.int64)
    event_buffer_y = np.array([], dtype=np.int64)
    event_buffer_t = np.array([], dtype=np.int64)
    aois = []
    last_aoi_update_us = None
    rpm_history: dict[int, list[float]] = {}
    frame_count = 0

    # Setup visualization window
    if not args.no_display:
        cv2.namedWindow("RPM Detection", cv2.WINDOW_NORMAL)

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
        max_window = max(AOI_WINDOW_US, RPM_WINDOW_US)
        event_buffer_x, event_buffer_y, event_buffer_t = trim_event_buffer(
            event_buffer_x, event_buffer_y, event_buffer_t, max_window
        )

        # Update AOIs periodically with heatmap/labels for logging
        new_aois, new_update_time, heatmap, labels = update_aois_if_needed(
            event_buffer_x,
            event_buffer_y,
            event_buffer_t,
            last_aoi_update_us,
            args.num_clusters,
        )
        if len(new_aois) > 0:
            aois = new_aois
            last_aoi_update_us = new_update_time

            # Log AOI detection
            if log_ctx:
                centroids = [aoi.centroid for aoi in aois]
                log_aoi_detection(log_ctx.logger, frame_count, len(aois), centroids)

        # Estimate RPM for each AOI
        rpm_estimates = estimate_aoi_rpms(
            event_buffer_x,
            event_buffer_y,
            event_buffer_t,
            aois,
            RPM_WINDOW_US,
            rpm_config,
        )

        # Log RPM estimates
        if log_ctx:
            for aoi_idx, rpm in rpm_estimates.items():
                log_rpm_estimate(log_ctx.logger, frame_count, aoi_idx, rpm)

        # Update RPM history and compute averages
        rpm_history = update_rpm_history(rpm_history, rpm_estimates)
        avg_rpms = compute_average_rpms(rpm_history)

        # Draw overlays on frame (for both display and logging)
        draw_aoi_bboxes(frame, aois, rpm_estimates, avg_rpms)

        # Save artifacts periodically
        if log_ctx and frame_count % LOG_EVERY_N_FRAMES == 0:
            if ENABLE_FRAME_LOGGING:
                save_frame(frame, frame_count, log_ctx.frames_dir)
            if ENABLE_HEATMAP_LOGGING and heatmap is not None:
                save_heatmap(heatmap, frame_count, log_ctx.heatmaps_dir)
            if ENABLE_LABEL_LOGGING and labels is not None:
                save_labels(labels, frame_count, log_ctx.labels_dir)

        # Visualize frame
        if not args.no_display:
            cv2.imshow("RPM Detection", frame)

            # Check for quit key (ESC or 'q')
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("\n\nUser requested quit.")
                break

        frame_count += 1

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

    # Cleanup
    if not args.no_display:
        cv2.destroyAllWindows()

    wall_time_total = time.perf_counter() - start_time

    print("\n\n--- Playback Finished ---")

    # Display final average RPMs
    if rpm_history:
        print("\n--- Final Average RPMs ---")
        final_avgs = compute_average_rpms(rpm_history)
        for aoi_idx in sorted(final_avgs.keys()):
            avg_rpm = final_avgs[aoi_idx]
            sample_count = len(rpm_history[aoi_idx])
            print(f"  AOI {aoi_idx}: {avg_rpm:.2f} RPM (n={sample_count} measurements)")

            # Log final averages
            if log_ctx:
                log_ctx.logger.info(
                    f"Final AOI {aoi_idx}: avg={avg_rpm:.2f} RPM (n={sample_count})"
                )
    else:
        print("\nNo RPM measurements recorded.")

    # Finalize logging
    if log_ctx:
        finalize_run(log_ctx, frame_count, wall_time_total)


if __name__ == "__main__":
    main()
