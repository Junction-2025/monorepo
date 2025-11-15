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
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # zeros -> white (255,255,255)
    mask_zero = arr == 0
    out[mask_zero] = [255, 255, 255]

    # -1 -> pure red (255,0,0)
    mask_neg1 = arr == -1
    out[mask_neg1] = [255, 0, 0]

    # scale all other (finite) values to 0..255 but reversed: min->255 and max->0
    mask_other = ~(mask_zero | mask_neg1) & np.isfinite(arr)
    if np.any(mask_other):
        vals = arr[mask_other].astype(np.float64)
        vmin = vals.min()
        vmax = vals.max()
        if vmax == vmin:
            # single value -> map to 0 (reversed)
            scaled = np.full(vals.shape, 0, dtype=np.uint8)
        else:
            # normalize to 0..255 then invert so min->255 and max->0
            norm = np.clip((vals - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            scaled = (255 - norm).astype(np.uint8)
        out[mask_other] = np.stack([scaled, scaled, scaled], axis=-1)

    Image.fromarray(out).save(LOG_DIR / f"{name}.png", format="PNG")
    print(f"--- Drew {LOG_DIR / f'{name}.png'} ---")
