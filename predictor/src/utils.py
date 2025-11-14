import numpy as np
from PIL import Image
from src.config import LOG_DIR
from src.yolo.yolo import detect_drone_crop
import cv2


def draw_png(arr: np.ndarray):
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
    
