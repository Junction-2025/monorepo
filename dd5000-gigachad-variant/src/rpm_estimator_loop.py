import argparse  # noqa: INP001

import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.evio_lib.pacer import Pacer
from src.evio_lib.dat_file import DatFileSource


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


def estimate_rpm_from_event_frames(frames, fps, blade_count):
    """
    frames: list of np.ndarray (gray scale intensity frames generated from event windows)
    fps: effective frame rate (1 / window_size)

    blade_count: number of blades on the propeller
    """
    if len(frames) < 20:
        print("Not enough frames for FFT.")
        return None

    # Convert frames to intensity values (1D array)
    intensity = []
    for frame in frames:
        # Mean intensity (you can switch to black counting if needed)
        intensity_value = np.mean(frame)
        intensity.append(intensity_value)

    intensity = np.array(intensity, dtype=float)
    # Remove DC bias
    signal = intensity - np.mean(intensity)

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
    parser.add_argument("--speed", type=float, default=0.1, help="Playback speed (1 is real time)")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed by dropping windows")
    args = parser.parse_args()

    ### HARDCODED VALUES
    # Hardcoded roi drone_idle (bounding box)
    roi = (500, 700, 300, 450)
    # Hardcoded roi fan_const_rpm (bounding box)
    # roi = (550, 700, 250, 440)
    # Hardcoded roi drone_moving (bounding box)
    #roi = (720, 920, 200, 330)
    blade_count = 2

    print("Loading data...")

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    print("Adding the frames to the list")

    roi_frames = []
    fps = 1000.0 / args.window  # window duration in ms
    max_frames = 100
    
    for batch_range in pacer.pace(src.ranges()):
        if len(roi_frames) >= max_frames:
            break
        window = get_window(
            src.event_words, src.order, batch_range.start, batch_range.stop
        )
        frame = get_frame(window)
        x_min, x_max, y_min, y_max = roi
        roi_frame = frame[y_min:y_max, x_min:x_max]
        roi_frames.append(roi_frame)

        # Show the ROI video
        cv2.imshow("ROI Video", roi_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Quit on 'q' or ESC
            break
        
    print("Frames loaded")

    print("Calculating RPM...")
    rpm, freqs, fft_vals = estimate_rpm_from_event_frames(roi_frames, fps, blade_count)

    print(f"Estimated RPM: {rpm:.2f}")
    print("RPM calculation complete")

    plt.plot(freqs, fft_vals)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT of Signal")
    plt.show()


if __name__ == "__main__":
    main()
