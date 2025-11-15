import argparse  # noqa: INP001
import time

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    crop_x: tuple[int, int] | None = None,
    crop_y: tuple[int, int] | None = None,
    base_color: tuple[int, int, int] = (127, 127, 127),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    
    # Adjust frame size if cropping
    if crop_x is not None:
        x_min, x_max = crop_x
        width = x_max - x_min
        x_coords = x_coords - x_min  # Offset coordinates
    if crop_y is not None:
        y_min, y_max = crop_y
        height = y_max - y_min
        y_coords = y_coords - y_min  # Offset coordinates
    
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file",
                        default="/Users/jessesorsa/Business/Junction_2025/project/monorepo/data/drone_idle.dat")
    parser.add_argument(
        "--window", type=float, default=0.2, help="Windows duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--crop-x",
        type=int,
        nargs=2,
        metavar=("X_MIN", "X_MAX"),
        default=None,
        help="Crop X coordinates (e.g., --crop-x 200 800)",
    )
    parser.add_argument(
        "--crop-y",
        type=int,
        nargs=2,
        metavar=("Y_MIN", "Y_MAX"),
        default=None,
        help="Crop Y coordinates (e.g., --crop-y 100 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (e.g., --output output.mp4)",
    )
    args = parser.parse_args()

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    # Enforce playback speed via dropping:
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # Setup video writer if output is specified
    video_writer = None
    if args.output:
        # Calculate FPS from window duration (or use provided FPS)
        fps = (1000.0 / args.window)  # frames per second
        
        print("THE FPS: ", fps)
        
        # Get frame dimensions
        width = 1280
        height = 720
        
        # Use codec based on file extension
        if args.output.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Writing video to {args.output} at {fps:.2f} FPS ({width}x{height})")
    
    batch_count = 0
    max_batches = 1000
    
    for batch_range in pacer.pace(src.ranges()):
        
        batch_count += 1
        if batch_count > max_batches:
            break
        
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        frame = get_frame(window)
        draw_hud(frame, pacer, batch_range)

        # Write frame to video if writer is set up
        if video_writer:
            video_writer.write(frame)

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    # Cleanup: release video writer
    if video_writer:
        video_writer.release()
        print(f"Video saved to {args.output}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
