from pathlib import Path

BATCH_WINDOW_US = 1000
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_NAME = "best-final.pt"

BASE_WIDTH = 1280
BASE_HEIGHT = 720

MODEL_LOGGING_VERBOSE = True

YOLO_CONFIDENCE_THRESHOLD = 0.01
HEATMAP_PIXEL_SIZE = 4

# Balanced threshold for propeller center detection
CENTROID_EPSILON = 0.65

# Maximum number of centroids (propeller centers) to detect
MAX_CENTROIDS = 6

K_CANDIDATES = [1, 2, 3, 4]

# More lenient outlier filtering
OUTLIER_DISTANCE_MULTIPLIER = 5.0

LOWER_RPM_BOUND = 1001
