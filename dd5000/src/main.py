import argparse
from pathlib import Path

import numpy as np

from src.config import BATCH_WINDOW_US, BASE_WIDTH, BASE_HEIGHT

from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource
from src.evio_lib.play_dat import get_frame, get_window

from src.yolo import detect_drone_crop
from src.logger import get_logger
from src.kmeans import get_blade_count, get_propeller_masks
from src.utils import overlay_mask
import cv2


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
        help="Disable video display (for testing).",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable logging output (for testing).",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="Number of clusters for K-means (default: 1).",
    )
    parser.add_argument(
        "--symmetry",
        type=int,
        default=3,
        help="Symmetry parameter (default: 3).",
    )

    return parser.parse_args()


def extract_roi_intensity(window, roi):
    """Extract event intensity from a region of interest.

    Args:
        window: Tuple of (x_coords, y_coords, polarities)
        roi: Tuple of (x1, x2, y1, y2) defining the bounding box

    Returns:
        Number of ON events in the ROI
    """
    x, y, pol = window
    x1, x2, y1, y2 = roi

    mask = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)

    # Return number of ON events
    return np.sum(pol[mask])


def estimate_rpm_from_signals(signals, fps, blade_count):
    """Estimate RPM from intensity signals using FFT analysis.

    Args:
        signals: List of intensity values over time
        fps: Effective signal rate (1 / window_size)
        blade_count: Number of blades on the propeller

    Returns:
        Tuple of (rpm, freqs, fft_vals) or None if not enough signals
    """
    if len(signals) < 20:
        return None

    # Remove DC bias
    signals = np.array(signals, dtype=float)
    signal = signals - np.mean(signals)

    # FFT
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / fps)

    # Ignore the zero-frequency peak
    peak_idx = np.argmax(fft_vals[1:]) + 1
    rot_freq_hz = freqs[peak_idx]

    # Calculate RPM
    rpm = (rot_freq_hz * 60) / blade_count

    return rpm, freqs, fft_vals


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

    logger = get_logger() if not args.no_logging else None

    window_name = "Drone Detection"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # RPM estimation variables
    roi_signals = []
    current_rpm = None
    rpm_measurements = []  # Track all RPM measurements for averaging
    fps = 1000000.0 / args.window  # Convert microseconds to Hz
    max_signals = 100  # Number of samples before running FFT
    last_blade_count = 0  # Track last known blade count for final calculation

    if logger:
        logger.info("\n--- Data Import Complete ---")
    for batch_range in pacer.pace(src.ranges()):
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

            blade_count = get_blade_count(cropped_frame, mask)
            avg_blade_count = int(sum([b[0] for b in blade_count]) / (len(blade_count) or 1))
            if avg_blade_count > 0:
                last_blade_count = avg_blade_count  # Track for final calculation

            # Extract ROI intensity for RPM estimation
            roi = (
                int(yolo_bounding_box.x1),
                int(yolo_bounding_box.x2),
                int(yolo_bounding_box.y1),
                int(yolo_bounding_box.y2),
            )
            event_intensity = extract_roi_intensity(window, roi)
            roi_signals.append(event_intensity)

            # Estimate RPM every max_signals frames
            if len(roi_signals) >= max_signals and avg_blade_count > 0:
                rpm_result = estimate_rpm_from_signals(roi_signals, fps, avg_blade_count)
                if rpm_result:
                    current_rpm = rpm_result[0]
                    rpm_measurements.append(current_rpm)
                    if logger:
                        logger.info(f"Estimated RPM: {current_rpm:.2f}")
                roi_signals.clear()

            # Display blade count and RPM
            if not args.no_display:
                text_pos = (tl[0], max(0, tl[1] - 8))
                text_content = f"Blades (AVG): {avg_blade_count}"
                if current_rpm is not None:
                    text_content += f" | RPM: {current_rpm:.1f}"

                cv2.putText(
                    frame,
                    text_content,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if not args.no_display:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

    # Process any remaining signals at the end
    if roi_signals and len(roi_signals) >= 20 and last_blade_count > 0:
        rpm_result = estimate_rpm_from_signals(roi_signals, fps, last_blade_count)
        if rpm_result:
            current_rpm = rpm_result[0]
            rpm_measurements.append(current_rpm)
            if logger:
                logger.info(f"Final Estimated RPM: {current_rpm:.2f} (from {len(roi_signals)} signals)")

    # Output final RPM statistics for test parsing
    if rpm_measurements:
        avg_rpm = sum(rpm_measurements) / len(rpm_measurements)
        print(f"AOI 0: {avg_rpm:.2f} RPM (n={len(rpm_measurements)})")
    else:
        if not args.no_logging:
            print(f"DEBUG: No RPM measurements - collected {len(roi_signals)} signals, last_blade_count={last_blade_count}")

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
