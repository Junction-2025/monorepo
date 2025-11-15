import numpy as np
from PIL import Image
from src.config import LOG_DIR
import cv2
import colorsys
from src.config import K_CANDIDATES


def draw_png(arr: np.ndarray, name="frame"):
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # random unique number to avoid overwriting
    num = np.random.randint(0, 1_000_000)
    Image.fromarray(arr).save(f"drones/frame-{num}.png", format="PNG")


def draw_labels(arr: np.ndarray, name="labels"):
    """
    Assume arr is a 1D flattened label array of length width*height and coerce to (height, width).
    Assign each unique label a deterministic color using evenly spaced HSV hues.
    """
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    labels = np.unique(arr)
    labels.sort()
    n = len(labels) or 1

    for i, label in enumerate(labels):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        color = (int(r * 255), int(g * 255), int(b * 255))
        out[arr == label] = color

    Image.fromarray(out).save(LOG_DIR / f"{name}.png", format="PNG")
    print(f"--- Drew {LOG_DIR / f'{name}.png'} ---")


def display_frame(frame: np.ndarray):
    """
    Display frame using OpenCV's imshow.
    Press any key to close the window.
    """
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay colored mask on frame with max(K_CANDIDATES) colors."""
    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
    ][: max(K_CANDIDATES)]

    overlay = frame.copy()
    for label in range(max(K_CANDIDATES)):
        overlay[mask == label] = colors[label]

    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
