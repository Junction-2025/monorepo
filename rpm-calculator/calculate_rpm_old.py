import numpy as np
import matplotlib.pyplot as plt


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
