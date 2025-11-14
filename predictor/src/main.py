import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Imports are now relative within the 'evio' package
from src.evio_in.pacer import Pacer
from src.evio_in.dat_file import DatFileSource, BatchRange


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stream a .dat file with real-time pacing and optional GUI.")
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
        "--window-us",
        type=int,
        default=10000,
        help="Batch window size in microseconds (default: 10000 µs = 10ms).",
    )
    # Add GUI control argument
    parser.add_argument(
        "--gui",
        action="store_false",
        help="Disable the live GUI visualization.",
    )
    return parser.parse_args()

def decode_batch(source: DatFileSource, batch: BatchRange) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the event data for a given batch range."""
    # Get the time-ordered indices for the current batch
    ordered_indices = source.order[batch.start:batch.stop]
    
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
    print(f"GUI Enabled: {not args.gui}")
    print("---------------------")

    try:
        source = DatFileSource(path=str(args.input), window_length_us=args.window_us)
        print(f"Source loaded with {len(source)} batches.")

        pacer = Pacer(speed=args.speed, force_speed=args.force_speed, drop_tolerance_s=args.drop_tolerance)
        
        raw_batches = source.ranges()
        paced_batches = pacer.pace(raw_batches)

        print("\n--- Starting Paced Playback ---")
        start_time = time.perf_counter()

        # --- GUI Setup ---
        if not args.gui:
            plt.ion()
            fig, ax = plt.subplots()
            
            # Create two scatter plots for different polarities
            scatter_on = ax.scatter([], [], c='blue', s=2, label='ON')
            scatter_off = ax.scatter([], [], c='red', s=2, label='OFF')
            
            ax.set_xlim(0, source.width)
            ax.set_ylim(source.height, 0) # Invert Y axis to match sensor origin
            ax.set_aspect('equal', adjustable='box')
            ax.set_title("Live Event Stream")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            
            # Create a legend
            on_patch = mpatches.Patch(color='blue', label='ON polarity')
            off_patch = mpatches.Patch(color='red', label='OFF polarity')
            ax.legend(handles=[on_patch, off_patch])

            fig.show()
        # --- End GUI Setup ---

        for batch in paced_batches:
            if not args.gui:
                x, y, pol = decode_batch(source, batch)
                
                # Separate points by polarity
                on_mask = pol == 1
                off_mask = pol == 0
                
                # Update scatter plot data
                scatter_on.set_offsets(np.c_[x[on_mask], y[on_mask]])
                scatter_off.set_offsets(np.c_[x[off_mask], y[off_mask]])
                
                ax.set_title(f"Live Event Stream | Event Time: {batch.end_ts_us / 1e6:.2f}s")
                
                # Redraw canvas
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                # Original text-based output
                wall_time = time.perf_counter() - start_time
                print(
                    f"\rWall Time: {wall_time: >6.2f}s | "
                    f"Event Time: {batch.end_ts_us / 1e6: >6.2f}s | "
                    f"Emitted: {pacer.emitted_batches: >5} | "
                    f"Dropped: {pacer.dropped_batches: >5}   ",
                    end=""
                )

        print("\n\n--- Playback Finished ---")
        
        if not args.gui:
            print("Closing GUI window.")
            plt.ioff()
            plt.close(fig)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Make sure plot is closed on error
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        sys.exit(1)


if __name__ == "__main__":
    main()