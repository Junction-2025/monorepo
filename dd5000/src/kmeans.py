import numpy as np
from src.config import HEATMAP_PIXEL_SIZE
from src.logger import get_logger
from dataclasses import dataclass

logger = get_logger()

def construct_heatmap(frame: np.ndarray, factor: int) -> np.ndarray:
    x, y = frame[0], frame[1]
    width, height = len(x), len(y)

    heatmap = np.zeros((height, width), dtype=np.int32)
    np.add.at(heatmap, (x, y), 1)

    h_new = height // factor
    w_new = width // factor
    reduced = (
        heatmap[: h_new * factor, : w_new * factor]
        .reshape(h_new, factor, w_new, factor)
        .sum(axis=(1, 3))
    )

    return reduced

@dataclass(frozen=True)
class Centroids():
    x_coords: np.ndarray
    y_coords: np.ndarray

def get_heatmap_centroids(heatmap: np.ndarray) -> :


def get_propeller_masks(frame: np.ndarray) -> np.ndarray:
    heatmap = construct_heatmap(frame, factor=HEATMAP_PIXEL_SIZE)
    logger.info(f"Constructed heatmap: {heatmap} of dim {heatmap.shape}")
    centroids = get_heatmap_centroids(heatmap)



    return np.array([])

def get_blade_count() -> int:
    return 1