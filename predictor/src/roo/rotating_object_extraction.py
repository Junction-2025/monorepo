import numpy as np
from typing import List

from src.config import DEFAULT_HEIGHT, DEFAULT_WIDTH
from src.roo.config import HEATMAP_PIXEL_SIZE, USE_ADAPTIVE_K, DEFAULT_K
from src.roo.kmeans import locate_centroids
from src.yolo.yolo import CropCoords
from src.utils import draw_heatmap, draw_labels
from numpy.typing import NDArray


def find_heatmap(
    x: NDArray[np.int64],
    y: NDArray[np.int64],
    height: int,
    width: int,
    factor: int = HEATMAP_PIXEL_SIZE,
    crop: CropCoords | None = None,
) -> NDArray[np.int64]:
    # apply crop if provided
    if crop is not None:
        valid_crop = (x >= crop.x1) & (x < crop.x2) & (y >= crop.y1) & (y < crop.y2)
        x = x[valid_crop] - crop.x1
        y = y[valid_crop] - crop.y1
        width = crop.x2 - crop.x1
        height = crop.y2 - crop.y1

    heatmap = np.zeros((height, width), dtype=np.int32)
    ix = x.astype(np.intp)
    iy = y.astype(np.intp)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    if np.any(valid):
        np.add.at(heatmap, (iy[valid], ix[valid]), 1)

    # reduce dimensions
    h_new = height // factor
    w_new = width // factor
    reduced = (
        heatmap[: h_new * factor, : w_new * factor]
        .reshape(h_new, factor, w_new, factor)
        .sum(axis=(1, 3))
    )

    return reduced


def find_clusters(
    x: np.ndarray,
    y: np.ndarray,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    drone_crop_coords: CropCoords | None = None,
    k: int = DEFAULT_K,
    use_adaptive_k: bool = USE_ADAPTIVE_K,
    k_candidates: List[int] | None = None,
) -> np.ndarray:
    """
    Find centroids in event data using heatmap-based initialization and k-means.

    Args:
        x: X coordinates of events
        y: Y coordinates of events
        height: Frame height
        width: Frame width
        drone_crop_coords: Optional crop coordinates for drone region
        k: Number of clusters (used when adaptive selection disabled)
        use_adaptive_k: Enable adaptive K selection using Davies-Bouldin Index
        k_candidates: List of K values to evaluate for adaptive selection

    Returns:
        labels: Label map with cluster assignments for each pixel
    """

    heatmap = find_heatmap(x, y, width=width, height=height, crop=drone_crop_coords)

    labels, cx, cy = locate_centroids(
        heatmap,
        k=k,
        use_adaptive_k=use_adaptive_k,
        k_candidates=k_candidates,
    )
    print(f"Centroids: ({cx}, {cy})")
    print(f"Heatmap: {heatmap.shape}")
    print(f"Labels: {labels}, unique = {np.unique(labels)}, dim = {labels.shape}")

    # === TEST CODE ===
    # mark centroid locations in the reduced heatmap as -1
    ix = np.rint(cx).astype(np.intp)
    iy = np.rint(cy).astype(np.intp)
    h_new, w_new = heatmap.shape
    valid = (ix >= 0) & (ix < w_new) & (iy >= 0) & (iy < h_new)
    heatmap[iy[valid], ix[valid]] = -1
    draw_heatmap(heatmap, name="heatmap")
    draw_labels(labels)
    # === TEST CODE ===

    return labels
