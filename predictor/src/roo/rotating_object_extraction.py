import numpy as np

from src.config import DEFAULT_HEIGHT, DEFAULT_WIDTH, LOG_DIR
from src.roo.config import HEATMAP_PIXEL_SIZE
from src.roo.kmeans import locate_centroids
from src.yolo.yolo import CropCoords
from src.utils import draw_heatmap
from numpy.typing import NDArray


def find_heatmap(
    x: NDArray[np.int_],
    y: NDArray[np.int_],
    height: int,
    width: int,
    factor: int = HEATMAP_PIXEL_SIZE,
    crop: CropCoords | None = None,
) -> NDArray[np.int_]:
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
) -> np.ndarray:
    """
    Find centroids in event data using heatmap-based initialization and k-means.

    Args:
        x: X coordinates of events
        y: Y coordinates of events
        cfg: Configuration for clustering

    Returns:
        mask: mask with boolean coords that is overlayed on data
    """

    heatmap = find_heatmap(x, y, width=width, height=height, crop=drone_crop_coords)
    np.savetxt(str(LOG_DIR / "heatmap.txt"), heatmap, fmt="%d")

    cx, cy = locate_centroids(heatmap)

    # mark centroid locations in the reduced heatmap as -1
    if cx is not None and cy is not None and len(cx) and len(cy):
        h_new, w_new = heatmap.shape
        ix = np.rint(cx).astype(np.intp)
        iy = np.rint(cy).astype(np.intp)
        valid = (ix >= 0) & (ix < w_new) & (iy >= 0) & (iy < h_new)
        if np.any(valid):
            heatmap[iy[valid], ix[valid]] = -1
    draw_heatmap(heatmap, name="heatmap")

    return heatmap
