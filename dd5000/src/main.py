import argparse
from pathlib import Path
import time

from src.config import BATCH_WINDOW_US, BASE_WIDTH, BASE_HEIGHT, LOWER_RPM_BOUND

from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource
from src.evio_lib.play_dat import get_frame, get_window

from src.yolo import detect_drone_crop
from src.logger import get_logger
from src.kmeans import get_blade_count, get_propeller_masks
from src.utils import overlay_mask
import cv2


import numpy as np

logger = get_logger()


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

    return parser.parse_args()


def extract_roi_intensity(window, roi):
    x, y, pol = window
    x1, x2, y1, y2 = roi

    mask = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)

    on_count = np.sum(pol[mask])
    return on_count


def estimate_rpm_from_signals(signals, fps, blade_count):
    """
    signals: list of np.ndarray (gray scale intensity frames generated from event windows)
    fps: effective signal rate (1 / window_size)

    blade_count: number of blades on the propeller
    """
    if len(signals) < 20:
        logger.info("Not enough signals for FFT.")
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

    # Set your blade count
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

    logger = get_logger()

    window_name = "Drone Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    timings = {
        "get_data": [],
        "yolo": [],
        "post_processing": [],
        "per_iter": [],
    }

    roi = (500, 700, 230, 300)
    roi_signals = []
    fps = 1000000 / args.window
    max_signals = 100
    blade_count = 2 # Assume 2
    
    frame_counter = 0
    rpm_records = []

    logger.info("\n--- Data Import Complete ---")
    for batch_range in pacer.pace(src.ranges()):
        frame_counter += 1
        
        window = get_window(
            src.event_words, src.order, batch_range.start, batch_range.stop
        )
        event_intensity = extract_roi_intensity(window, roi)
        roi_signals.append(event_intensity)

        # Run FFT every N frames
        if len(roi_signals) >= max_signals:
            output = estimate_rpm_from_signals(
                roi_signals, fps, blade_count
            )
            assert output is not None
            rpm, _, _ = output
            rpm_records.append(rpm)
            logger.info(f"RPM: {rpm:.2f}")
            roi_signals.clear()        
        
        if frame_counter % 30 == 0:
            t0 = time.perf_counter()
            window = get_window(
                src.event_words,
                src.order,
                batch_range.start,
                batch_range.stop,
            )
            frame = get_frame(window)
            t1 = time.perf_counter()
            timings["get_data"].append(t1 - t0)
            t2 = time.perf_counter()
            yolo_bounding_box = detect_drone_crop(frame)
            t3 = time.perf_counter()
            timings["yolo"].append(t3 - t2)

            t4 = time.perf_counter()
            if yolo_bounding_box:
                tl = (int(yolo_bounding_box.x1), int(yolo_bounding_box.y1))
                br = (int(yolo_bounding_box.x2), int(yolo_bounding_box.y2))
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                
                roi = (yolo_bounding_box.x1, yolo_bounding_box.x2, yolo_bounding_box.y1, yolo_bounding_box.y2)

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

                blade_count_array = get_blade_count(cropped_frame, mask)
                text_pos = (tl[0], max(0, tl[1] - 8))
                cv2.putText(
                    frame,
                    "".join(
                        [
                            f"Blades (AVG): {int(sum([b[0] for b in blade_count_array]) / (len(blade_count_array) or 1))}"
                        ]
                    ),
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                )
            t5 = time.perf_counter()
            timings["post_processing"].append(t5 - t4)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    def _profiling(name):
        times = timings[name]
        if not times:
            return
        avg = sum(times) / len(times)
        logger.info(
            f"{name:>10}: "
            f"avg={avg * 1000:.2f} ms, "
            f"min={min(times) * 1000:.2f} ms, "
            f"max={max(times) * 1000:.2f} ms, "
            f"n={len(times)}"
        )

    logger.info("\n=== AVERAGE RPM ===")
    rpm_records = [r for r in rpm_records if r >= LOWER_RPM_BOUND]
    if rpm_records:
        avg_rpm = float(sum(rpm_records)) / len(rpm_records)
        logger.info("RPM: %.2f", avg_rpm)
    else:
        logger.info("No RPM records above LOWER_RPM_BOUND (%s)", LOWER_RPM_BOUND)

    logger.info("\n=== Timing stats ===")
    for key in timings:
        _profiling(key)


if __name__ == "__main__":
    main()
