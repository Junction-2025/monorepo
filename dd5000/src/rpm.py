import numpy as np
from src.logger import get_logger

logger = get_logger()

def extract_roi_intensity(window, roi):
    x, y, pol = window
    x1, x2, y1, y2 = roi

    mask = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)

    on_count = np.sum(pol[mask])
    return on_count


def estimate_rpm_from_signals(signals, fps, blade_count):
    """
    signals: list of np.ndarray (gray scale intensity frames generated from event windows)
    fps: effective signal rate (1 / window_size)

    blade_count: number of blades on the propeller
    """
    if len(signals) < 20:
        logger.info("Not enough signals for FFT.")
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
