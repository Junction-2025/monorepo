import argparse
from pathlib import Path
import time

from src.config import BATCH_WINDOW_US, BASE_WIDTH, BASE_HEIGHT, LOWER_RPM_BOUND
from src.logger import get_logger
from src.profiling import create_timings, add_timing, log_timings, timing_section


from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource
from src.evio_lib.play_dat import get_frame, get_window

from src.yolo import detect_drone_crop
from src.kmeans import get_blade_count, get_propeller_masks
from src.utils import overlay_mask
from src.rpm import extract_roi_intensity, estimate_rpm_from_signals
import cv2


import numpy as np

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
        with timing_section(timings, "extract_roi_intensity"):
            event_intensity = extract_roi_intensity(window, roi)

        roi_signals.append(event_intensity)

        # Run FFT every N frames
        if len(roi_signals) >= max_signals:
            with timing_section(timings, "estimate_rpm_from_signal"):
                output = estimate_rpm_from_signals(
                    roi_signals, fps, blade_count
                )
            assert output is not None
            rpm, _, _ = output
            rpm_records.append(rpm)
            logger.info(f"RPM: {rpm:.2f}")
            roi_signals.clear()        
        
        if frame_counter % 30 == 0:
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
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
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
