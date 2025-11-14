from ultralytics import YOLOE
import numpy as np
import cv2
from pathlib import Path

# Load model once globally
model = YOLOE("yolo12l.pt")
# model.set_classes(["drone"], model.get_text_pe(["drone"]))

def detect_drone_crop(frame: np.ndarray):
    """
    Input:  frame (H,W,3) BGR ndarray
    Output: cropped drone ndarray OR None
    """
    result = model.predict(frame)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    # take first or highest-confidence box
    b = result.boxes[0]
    x1, y1, x2, y2 = map(int, b.xyxy[0])

    # crop
    crop = frame[y1:y2, x1:x2]
    return crop

