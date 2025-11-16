import argparse  # noqa: INP001

import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.profiling import create_timings, add_timing, log_timings, timing_section
from src.logger import get_logger

from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource

timings = create_timings()
logger = get_logger()

""""
Run this file by running:
uv run python src/.py /path/to/file.dat
"""

def get_window(
    event_words: np.ndarray, time_order: np.ndarray, win_start: int, win_stop: int
):
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, pixel_polarity


def get_frame(
    window,
    width=1280,
    height=720,
    *,
    base_color=(127, 127, 127),
    on_color=(255, 255, 255),
    off_color=(0, 0, 0),
):
    x_coords, y_coords, polarities_on = window

    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame

def extract_roi_intensity(window, roi):
    x, y, pol = window
    x1, x2, y1, y2 = roi

    mask = (
        (x >= x1) & (x < x2) &
        (y >= y1) & (y < y2)
    )

    on_count  = np.sum(pol[mask])
    off_count = np.sum(~pol[mask])
    #why only on_count?
    contrast = on_count - off_count
    return on_count


def estimate_rpm_from_signals(signals, fps, blade_count):
    """
    signals: list of np.ndarray (gray scale intensity frames generated from event windows)
    fps: effective signal rate (1 / window_size)

    blade_count: number of blades on the propeller
    """
    if len(signals) < 20:
        print("Not enough signals for FFT.")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    #parser.add_argument("--window", type=float, default=0.25, help="Window duration in ms")
    parser.add_argument("--window", type=float, default=1, help="Window duration in ms")
    parser.add_argument("--speed", type=float, default=1, help="Playback speed (1 is real time)")
    args = parser.parse_args()

    ### HARDCODED VALUES
    # Hardcoded roi drone_idle (bounding box)
    #roi = (500, 700, 230, 430)
    # Hardcoded roi fan_const_rpm (bounding box)
    # roi = (550, 700, 250, 440)
    # Hardcoded roi drone_moving (bounding box)
    roi = (720, 920, 200, 330)
    blade_count = 2

    print("Loading data...")

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )
    pacer = Pacer(speed=args.speed, force_speed=True)

    print("Adding the frames to the list")

    roi_signals = []
    fps = 1000.0 / args.window  # window duration in ms
    max_signals = 100
    
    frame_counter = 0
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
            with timing_section(timings, "estimate_rpm_from_signals"):
                rpm, freqs, fft_vals = estimate_rpm_from_signals(
                    roi_signals, fps, blade_count
                )


            print(f"RPM: {rpm:.2f}")
            roi_signals.clear()        
        
        if frame_counter % 30 == 0:
            frame = get_frame(window)
            # Draw ROI bounding box
            x1, x2, y1, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
            cv2.imshow("Events", frame)
            cv2.waitKey(1)

    log_timings(logger, timings, title="RPM benchmarks")
if __name__ == "__main__":
    main()
