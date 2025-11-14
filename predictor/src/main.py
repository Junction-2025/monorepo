import argparse
import sys
import time
from pathlib import Path

# Imports are now relative within the 'evio' package
from src.evio_in.pacer import Pacer
from src.evio_in.dat_file import DatFileSource, BatchRange


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stream a .dat file with real-time pacing.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True, # Made required as there's no default DATA_DIR in evio
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
        "--window-us",
        type=int,
        default=10000,
        help="Batch window size in microseconds (default: 10000 µs = 10ms).",
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
    print(f"Frame dropping: {'Enabled' if args.force_speed else 'Disabled'}")
    print(f"Drop tolerance: {args.drop_tolerance}s")
    print(f"Window size: {args.window_us} µs")
    print("---------------------")
    
    try:
        # 1. Create a data source from the .dat file
        source = DatFileSource(
            path=str(args.input),
            window_length_us=args.window_us
        )
        print(f"Source loaded with {len(source)} batches.")

        # 2. Initialize the Pacer with desired settings
        pacer = Pacer(
            speed=args.speed,
            force_speed=args.force_speed,
            drop_tolerance_s=args.drop_tolerance
        )

        # 3. Get the iterator of batches from the source
        raw_batches = source.ranges()

        # 4. Wrap the iterator with the Pacer
        paced_batches = pacer.pace(raw_batches)

        print("\n--- Starting Paced Playback ---")
        start_time = time.perf_counter()

        # 5. Loop over the paced batches
        for batch in paced_batches:
            # Simulate work by sleeping briefly
            time.sleep(0.001)

            # Print live stats on a single line
            wall_time = time.perf_counter() - start_time
            print(
                f"\rWall Time: {wall_time: >6.2f}s | "
                f"Event Time: {batch.end_ts_us / 1e6: >6.2f}s | "
                f"Emitted: {pacer.emitted_batches: >5} | "
                f"Dropped: {pacer.dropped_batches: >5}   ",
                end=""
            )

        print("\n\n--- Playback Finished ---")
        print(f"Total batches emitted: {pacer.emitted_batches}")
        print(f"Total batches dropped: {pacer.dropped_batches}")
        print(f"Average drop rate: {pacer.average_drop_rate:.2f} drops/(event-time ms)")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()