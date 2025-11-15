from ultralytics import YOLOWorld

from src.models import CropCoords
import numpy as np

from src.config import YOLO_CONFIDENCE_THRESHOLD
from src.logger import get_logger

# Load model once globally
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["drone", "uav", "quadrotor", "airplane"])

logger = get_logger()


def detect_drone_crop(frame: np.ndarray) -> CropCoords | None:
    """
    Input:  frame (H,W,3) BGR ndarray
    Output: cropped drone CropCoords OR None

    Args:
        frame: Input frame to detect drone in
        conf_threshold: Minimum confidence threshold (0-1), default 0.1 for event camera frames
        verbose: Whether to print detection info
    """
    result = model.predict(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        logger.debug("YOLO: No drone detected")
        return None

    # take first or highest-confidence box
    b = result.boxes[0]
    confidence = float(b.conf[0])
    x1, y1, x2, y2 = map(int, b.xyxy[0])

    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    logger.debug(
        f"YOLO: Detected drone at ({x1},{y1})-({x2},{y2}) conf={confidence:.3f}"
    )

    # crop
    return CropCoords(x1=x1, x2=x2, y1=y1, y2=y2)
