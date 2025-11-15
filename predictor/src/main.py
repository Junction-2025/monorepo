import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Imports are now relative within the 'evio' package
from src.evio_in.pacer import Pacer
from src.evio_in.dat_file import DatFileSource, BatchRange
from src.evio_in.play_dat import get_frame, get_window
from src.yolo.yolo import detect_drone_crop
from src.roo.rotating_object_extraction import find_clusters


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
        default=10000,
        help="Batch window size in microseconds (default: 10000 µs = 10ms).",
    )
    # The --gui argument is no longer needed without a GUI
    return parser.parse_args()


def decode_batch(
    source: DatFileSource, batch: BatchRange
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the event data for a given batch range."""
    # Get the time-ordered indices for the current batch
    ordered_indices = source.order[batch.start : batch.stop]

    # Get the corresponding packed event words
    w32 = source.event_words[ordered_indices].astype(np.uint32)

    # Decode polarity, x, and y from the packed words
    # Polarity: > 0 -> ON (1), 0 -> OFF (0)
    pol = ((w32 >> 28) & 0xF).astype(np.uint8)
    pol = (pol > 0).astype(np.uint8)
    y = ((w32 >> 14) & 0x3FFF).astype(np.int64)
    x = (w32 & 0x3FFF).astype(np.int64)

    return x, y, pol


def main():
    """Main streaming loop."""
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found at '{args.input}'")
        sys.exit(1)

    print("--- Configuration ---")
    print(f"Input file: {args.input}")
    print(f"Playback speed: {args.speed}x")
    print("---------------------")

    src = DatFileSource(
        args.input, width=1280, height=720, window_length_us=args.window * 15
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

    print("\n--- Data Import Complete ---")
    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        frame = get_frame(window)

        drone_crop_coords = detect_drone_crop(frame)
        if drone_crop_coords:
            find_clusters(window[0], window[1], drone_crop_coords=drone_crop_coords)

        wall_time = time.perf_counter() - start_time
        print(
            f"\rWall Time: {wall_time: >6.2f}s | "
            f"Event Time: {batch_range.end_ts_us / 1e6: >6.2f}s | "
            f"Emitted: {pacer.emitted_batches: >5} | "
            f"Dropped: {pacer.dropped_batches: >5}   ",
            end="",
        )

    print("\n\n--- Playback Finished ---")


if __name__ == "__main__":
    main()
