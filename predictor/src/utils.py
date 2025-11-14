import numpy as np
from PIL import Image
from src.config import LOG_DIR
import cv2


def draw_png(arr: np.ndarray, name="frame"):
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(LOG_DIR / "frame.png", format="PNG")


def display_frame(frame: np.ndarray):
    """
    Display frame using OpenCV's imshow.
    Press any key to close the window.
    """
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


def draw_heatmap(arr: np.ndarray, name="heatmap"):
    heatmap = np.asarray(arr)
    # zeros -> white (255), everything else -> black (0)
    mask = heatmap == 0
    out = np.where(mask, 255, 0).astype(np.uint8)

    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)

    Image.fromarray(out).save(LOG_DIR / f"{name}.png", format="PNG")
    print(f"--- Drew {LOG_DIR / f'{name}.png'} ---")
