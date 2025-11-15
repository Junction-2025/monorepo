import numpy as np
from src.config import HEATMAP_PIXEL_SIZE

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

def get_propeller_masks(frame: np.ndarray) -> np.ndarray:
    heatmap = construct_heatmap(frame, factor=HEATMAP_PIXEL_SIZE)
    return np.array([])

def get_blade_count() -> int:
    return 1