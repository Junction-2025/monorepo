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
    # work with a single-channel view (use first channel if given a 3-channel array)
    if heatmap.ndim == 2:
        scalar = heatmap
    else:
        scalar = heatmap[..., 0]

    h, w = scalar.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # zeros -> white (255,255,255)
    mask_zero = scalar == 0
    out[mask_zero] = [255, 255, 255]

    # -1 -> pure red (255,0,0)
    mask_neg1 = scalar == -1
    out[mask_neg1] = [255, 0, 0]

    Image.fromarray(out).save(LOG_DIR / f"{name}.png", format="PNG")
    print(f"--- Drew {LOG_DIR / f'{name}.png'} ---")
