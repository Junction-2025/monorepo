from ultralytics import YOLO

from src.models import CropCoords
import numpy as np

from src.config import YOLO_CONFIDENCE_THRESHOLD, MODEL_LOGGING_VERBOSE, MODEL_NAME
from src.logger import get_logger

# Load model once globally
model = YOLO(MODEL_NAME)  # Use standard YOLO, not YOLOWorld
logger = get_logger()


def detect_drone_crop(frame: np.ndarray) -> CropCoords | None:
    """
    Input:  frame (H,W,3) BGR ndarray
    Output: cropped drone CropCoords OR None

    Args:
        frame: Input frame to detect drone in
    """
    results = model.predict(
        frame,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=MODEL_LOGGING_VERBOSE,
        device="mps",
        iou=0.5,
    )

    if (
        not results
        or not hasattr(results[0], "boxes")
        or results[0].boxes is None
        or len(results[0].boxes) == 0
    ):
        logger.debug("YOLO: No drone detected")
        return None

    boxes = results[0].boxes

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
    except Exception:
        xyxy = np.array(boxes.xyxy)
        confs = np.array(boxes.conf)

    if xyxy.size == 0:
        logger.debug("YOLO: No drone extracted")
        return None

    # SIMPLE: choose the highest-confidence detection
    idx = int(np.argmax(confs))
    x1, y1, x2, y2 = map(int, xyxy[idx])
    confidence = float(confs[idx])

    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    logger.debug(f"YOLO: Detected drone ({x1},{y1})-({x2},{y2}) conf={confidence:.3f}")

    return CropCoords(x1=x1, x2=x2, y1=y1, y2=y2)
