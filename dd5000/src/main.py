import argparse
from pathlib import Path
import time

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

    logger.info("\n--- Data Import Complete ---")
    for batch_range in pacer.pace(src.ranges()):
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
        timings["yolo"].append(t3-t2)

        t4 = time.perf_counter()
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
            text_pos = (tl[0], max(0, tl[1] - 8))
            cv2.putText(
                frame,
                "".join(
                    [
                        f"Blades (AVG): {int(sum([b[0] for b in blade_count]) / (len(blade_count) or 1))}"
                    ]
                ),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
        t5 = time.perf_counter()
        timings["post_processing"].append(t5-t4)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    def profiling(name):
        times = timings[name]
        if not times:
            return
        avg = sum(times) / len(times)
        print(
            f"{name:>10}: "
            f"avg={avg*1000:.2f} ms, "
            f"min={min(times)*1000:.2f} ms, "
            f"max={max(times)*1000:.2f} ms, "
            f"n={len(times)}"
        )
        
    print("\n=== Timing stats ===")
    for key in timings:
        profiling(key)


if __name__ == "__main__":
    main()
