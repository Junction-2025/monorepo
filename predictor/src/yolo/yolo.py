from ultralytics import YOLOWorld
import numpy as np
from dataclasses import dataclass

# Load model once globally
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["drone", "uav", "quadrotor", "airplane"])


@dataclass
class CropCoords:
    x1: int
    x2: int
    y1: int
    y2: int


def detect_drone_crop(frame: np.ndarray) -> CropCoords | None:
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
    return CropCoords(x1=x1, x2=x2, y1=y1, y2=y2)
