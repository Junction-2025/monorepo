"""
Centralized configuration for RPM detection pipeline.

All configurable parameters are defined here for easy tuning and experimentation.
"""

from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Directory Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR.parent / "data"


# =============================================================================
# Sensor Configuration
# =============================================================================

DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280


# =============================================================================
# YOLO Detection
# =============================================================================

YOLO_MARGIN = 30  # Padding around detected drone region


# =============================================================================
# AOI Detection (Area of Interest / Rotating Object Extraction)
# =============================================================================

# Heatmap parameters
HEATMAP_PIXEL_SIZE = 4  # Bin size for spatial heatmap (pixels) - from paper
EPSILON = 0.3  # Threshold factor for centroid selection

# K-means clustering
DEFAULT_K = 4  # Default number of clusters when adaptive selection disabled
MAX_ITERATIONS = 100  # Maximum iterations for k-means convergence
CONVERGENCE_THRESHOLD = 1e-4  # Centroid movement threshold for convergence

# Adaptive K selection
USE_ADAPTIVE_K = True  # Enable automatic cluster count selection
K_CANDIDATES = [2, 3, 4, 5, 6]  # Candidate values to evaluate

# Outlier removal
OUTLIER_THRESHOLD_MULTIPLIER = 3.0  # Distance threshold factor (× median distance)

# AOI update timing
AOI_WINDOW_US = 150_000  # Time window for AOI detection (150ms)
AOI_UPDATE_INTERVAL_US = 50_000  # Update frequency (50ms)


# =============================================================================
# RPM Estimation
# =============================================================================


@dataclass
class RPMConfig:
    """Configuration for RPM estimation via correlation."""

    slice_us: int = 500  # Time slice duration for frame aggregation (µs)
    min_slices: int = 10  # Minimum slices needed for estimation
    symmetry_order: int = 3  # Blade/pattern repeats per revolution
    peak_prominence: float = 0.1  # Peak detection threshold (× std)
    max_peaks: int = 10  # Maximum peaks to consider


RPM_WINDOW_US = 30_000  # Time window for RPM estimation (30ms)


# =============================================================================
# Streaming / Buffering
# =============================================================================

BATCH_WINDOW_US = 100_000  # Default batch window size (150ms as per paper)
EVENT_BUFFER_MAX_US = 200_000  # Maximum event buffer retention (200ms)


# =============================================================================
# Visualization
# =============================================================================

BBOX_COLOR = (0, 255, 0)  # Green bounding boxes (BGR)
TEXT_COLOR = (0, 0, 255)  # Red RPM text (BGR)
CENTROID_COLOR = (255, 0, 0)  # Blue centroid dots (BGR)
TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2


# =============================================================================
# Logging
# =============================================================================

ENABLE_FRAME_LOGGING = True  # Save frames with overlays
ENABLE_HEATMAP_LOGGING = True  # Save heatmaps
ENABLE_LABEL_LOGGING = True  # Save cluster labels
LOG_EVERY_N_FRAMES = 3  # Save every Nth frame (to avoid excessive I/O)
