from pathlib import Path

BATCH_WINDOW_US = 150_000
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

BASE_WIDTH = 1280
BASE_HEIGHT = 720

YOLO_CONFIDENCE_THRESHOLD = 0.2
HEATMAP_PIXEL_SIZE = 4

CENTROID_EPSILON = 0.3
