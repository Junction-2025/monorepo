import argparse
from pathlib import Path

from src.config import BATCH_WINDOW_US, BASE_WIDTH, BASE_HEIGHT

from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource
from src.evio_lib.play_dat import get_frame, get_window

from src.logger import get_logger
from src.kmeans import get_blade_count, get_propeller_masks
import cv2
import numpy as np
from src.rpm_calculator import estimate_rpm_from_event_frames


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
        help=f"Batch window size in ms.",
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

    print("\n=== INITIALIZING DATA SOURCE ===")
    print(f"Input file: {args.input}")
    print(f"Window length (us): {args.window}")
    print(f"Speed: {args.speed}, Force speed: {args.force_speed}")
    print("--------------------------------")

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
    
    roi_frames = []
    fps = 1000000 / BATCH_WINDOW_US

    print(f"Computed FPS from event window: {fps:.2f}")

    logger.info("\n--- Data Import Complete ---")

    frame_index = 0

    for batch_range in pacer.pace(src.ranges()):
        frame_index += 1

        # Extract event window
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        # Convert events to a frame
        frame = get_frame(window)

        # Use fixed bounding box
        yolo_bounding_box = (500, 700, 250, 450)
        x1, x2, y1, y2 = yolo_bounding_box

        # Extract ROI frame
        roi_frame = frame[y1:y2, x1:x2]

        roi_frames.append(roi_frame)

        # Perform RPM estimation every 100 frames
        #if frame_index % 100 == 0:
        if frame_index == 100:
            
            print(f"Frame index: {frame_index}")
            
            print(f"ROI frames: {len(roi_frames)}")

            rpm, freqs, fft_vals = estimate_rpm_from_event_frames(
                roi_frames, fps, 2
            )
            print(f"Estimated RPM: {rpm}")

            roi_frames.clear()
            break
        
        # Draw bbox on display frame
        tl = (int(x1), int(y1))
        br = (int(x2), int(y2))
        cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
        
        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
        
    print("\n=== PROCESSING COMPLETE ===")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()