import argparse
from pathlib import Path

from src.config import BATCH_WINDOW_US, BASE_WIDTH, BASE_HEIGHT, LOWER_RPM_BOUND
from src.logger import get_logger
from src.profiling import create_timings, log_timings, timing_section


from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource
from src.evio_lib.play_dat import get_frame, get_window

from src.yolo import detect_drone_crop
from src.kmeans import get_blade_count, get_propeller_masks
from src.utils import overlay_mask
from src.rpm import extract_roi_intensity, estimate_rpm_from_signals
from src.blade_tracker import BladeCountTracker
import cv2

logger = get_logger()
timings = create_timings()


def parse_args() -> argparse.Namespace:
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
        "--force-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=BATCH_WINDOW_US,
        help=f"Batch window size in microseconds (default: {BATCH_WINDOW_US}µs).",
    )
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=0.1,
        help="Lag tolerance in seconds before dropping batches (default: 0.1s).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display for headless/benchmark mode.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    src = DatFileSource(
        args.input,
        width=BASE_WIDTH,
        height=BASE_HEIGHT,
        window_length_us=args.window,
    )
    pacer = Pacer(
        speed=args.speed,
        force_speed=args.force_speed,
        drop_tolerance_s=args.drop_tolerance,
    )

    logger = get_logger()

    # Setup display only if not in headless mode
    window_name = "Drone Detection"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    timings = {
        "get_data": [],
        "yolo": [],
        "post_processing": [],
        "per_iter": [],
    }

    # Use the same ROI as pure_rpm_calculations.py for drone_idle
    roi = (500, 700, 230, 430)
    roi_signals = []
    fps = 1000000 / args.window
    max_signals = 30  # Reduced for faster testing (was 100)

    # Initialize blade count tracker for KNN-based detection
    blade_tracker = BladeCountTracker(
        window_size=10,
        min_observations=3,
        default_blade_count=2,  # Fallback assumption
    )

    frame_counter = 0
    rpm_records = []
    current_rpm = 0.0
    running_avg_rpm = 0.0

    logger.info("\n--- Data Import Complete ---")
    for batch_range in pacer.pace(src.ranges()):
        frame_counter += 1

        window = get_window(
            src.event_words, src.order, batch_range.start, batch_range.stop
        )
        with timing_section(timings, "extract_roi_intensity"):
            event_intensity = extract_roi_intensity(window, roi)

        roi_signals.append(event_intensity)

        # Run FFT every N frames
        if len(roi_signals) >= max_signals:
            # Get current blade count estimate from tracker
            blade_count = blade_tracker.get_blade_count()
            tracker_stats = blade_tracker.get_stats()

            with timing_section(timings, "estimate_rpm_from_signal"):
                output = estimate_rpm_from_signals(roi_signals, fps, blade_count)
            assert output is not None
            rpm, _, _ = output
            rpm_records.append(rpm)
            current_rpm = rpm

            # Calculate running average from filtered records
            filtered_records = [r for r in rpm_records if r >= LOWER_RPM_BOUND]
            if filtered_records:
                running_avg_rpm = float(sum(filtered_records)) / len(filtered_records)

            # Log RPM with blade count context
            confidence_str = "confident" if tracker_stats["is_confident"] else "default"
            logger.info(
                f"RPM: {rpm:.2f} (using blade_count={blade_count}, {confidence_str}, "
                f"observations={tracker_stats['count']})"
            )
            roi_signals.clear()

        # Run YOLO/KNN detection periodically to update blade count
        # Run early and frequently to populate tracker before first RPM calculation
        if frame_counter % 10 == 0:
            window = get_window(
                src.event_words,
                src.order,
                batch_range.start,
                batch_range.stop,
            )
            frame = get_frame(window)

            yolo_bounding_box = detect_drone_crop(frame)

            if yolo_bounding_box:
                tl = (int(yolo_bounding_box.x1), int(yolo_bounding_box.y1))
                br = (int(yolo_bounding_box.x2), int(yolo_bounding_box.y2))
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)

                roi = (
                    yolo_bounding_box.x1,
                    yolo_bounding_box.x2,
                    yolo_bounding_box.y1,
                    yolo_bounding_box.y2,
                )

                cropped_frame = frame[
                    yolo_bounding_box.y1 : yolo_bounding_box.y2,
                    yolo_bounding_box.x1 : yolo_bounding_box.x2,
                ]
                mask = get_propeller_masks(cropped_frame)
                cropped_overlay = overlay_mask(cropped_frame, mask)
                frame[
                    yolo_bounding_box.y1 : yolo_bounding_box.y2,
                    yolo_bounding_box.x1 : yolo_bounding_box.x2,
                ] = cropped_overlay

                # Detect blade count using KNN and add to tracker
                blade_count_array = get_blade_count(cropped_frame, mask)
                blade_tracker.add_observation(blade_count_array)

                detected_avg = (
                    int(
                        sum([b[0] for b in blade_count_array])
                        / (len(blade_count_array) or 1)
                    )
                    if blade_count_array
                    else 0
                )
                current_estimate = blade_tracker.get_blade_count()

                text_pos = (tl[0], max(0, tl[1] - 8))
                cv2.putText(
                    frame,
                    f"Blades: detected={detected_avg}, estimate={current_estimate}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Display average RPM in red
                rpm_text_pos = (tl[0], max(0, tl[1] - 32))
                rpm_text = f"Avg RPM: {running_avg_rpm:.1f}"
                cv2.putText(
                    frame,
                    rpm_text,
                    rpm_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red color in BGR
                    2,
                )
            if not args.no_display:
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

    if not args.no_display:
        cv2.destroyAllWindows()

    log_timings(logger, timings, title="Main function benchmarks")

    logger.info("\n=== AVERAGE RPM ===")
    rpm_records = [r for r in rpm_records if r >= LOWER_RPM_BOUND]
    if rpm_records:
        avg_rpm = float(sum(rpm_records)) / len(rpm_records)
        logger.info("RPM: %.2f", avg_rpm)
    else:
        logger.info("No RPM records above LOWER_RPM_BOUND (%s)", LOWER_RPM_BOUND)


if __name__ == "__main__":
    main()
