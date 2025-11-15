import argparse
from pathlib import Path

from src.config import BATCH_WINDOW_US, BASE_WIDTH, BASE_HEIGHT

from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource
from src.evio_lib.play_dat import get_frame, get_window

from src.yolo import detect_drone_crop
from src.logger import get_logger
from src.kmeans import get_propeller_masks, get_blade_count
from src.utils import draw_labels
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
        if not yolo_bounding_box:
            continue

        cropped_frame = frame[
            yolo_bounding_box.y1 : yolo_bounding_box.y2,
            yolo_bounding_box.x1 : yolo_bounding_box.x2,
        ]
        #propeller_masks = get_propeller_masks(frame=cropped_frame)
        #draw_labels(propeller_masks)
        blade_count = get_blade_count()

        # Draw YOLO bbox and blade count on the frame
        tl = (int(yolo_bounding_box.x1), int(yolo_bounding_box.y1))
        br = (int(yolo_bounding_box.x2), int(yolo_bounding_box.y2))
        cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
        text_pos = (tl[0], max(0, tl[1] - 8))
        cv2.putText(
            frame,
            f"Blades: {blade_count}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Drone RPM", frame)
  


if __name__ == "__main__":
    main()
