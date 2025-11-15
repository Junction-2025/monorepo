import numpy as np
from src.config import HEATMAP_PIXEL_SIZE, CENTROID_EPSILON
from src.logger import get_logger
from src.models import Centroids

logger = get_logger()

def construct_heatmap(frame: np.ndarray, factor: int) -> np.ndarray:
    height, width = frame.shape
    h_new = height // factor
    w_new = width // factor
    reduced = (
        frame[: h_new * factor, : w_new * factor]
        .reshape(h_new, factor, w_new, factor)
        .sum(axis=(1, 3))
    )

    return reduced

def get_heatmap_centroids(heatmap: np.ndarray) -> Centroids:   
    heatmap_max_val = heatmap.max()
    centroid_inclusion_threshold = heatmap_max_val * CENTROID_EPSILON



    return Centroids(

    )



def get_propeller_masks(frame: np.ndarray) -> np.ndarray:
    heatmap = construct_heatmap(frame, factor=HEATMAP_PIXEL_SIZE)
    logger.info(f"Constructed heatmap: {heatmap} of dim {heatmap.shape}")
    centroids = get_heatmap_centroids(heatmap)



    return np.array([])

def get_blade_count() -> int:
    return 1