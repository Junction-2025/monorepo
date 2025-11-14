import numpy as np
from typing import Tuple
from numpy import ndarray

from src.config import DEFAULT_HEIGHT, DEFAULT_WIDTH, LOG_DIR
from src.roo.config import HEATMAP_PIXEL_SIZE
from src.roo.kmeans import kmeans_cluster
from src.utils import draw_heatmap, draw_png
import numpy as _np
from numpy.typing import NDArray


def find_heatmap(x: NDArray[np.int_], y: NDArray[np.int_], height: int, width: int, factor: int = 4) -> NDArray[np.int_]:
    heatmap = np.zeros((height, width), dtype=np.int32)
    ix = x.astype(np.intp)
    iy = y.astype(np.intp)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    if np.any(valid):
        np.add.at(heatmap, (iy[valid], ix[valid]), 1)
    
    h_new = height // factor
    w_new = width // factor
    reduced = heatmap[:h_new*factor, :w_new*factor].reshape(h_new, factor, w_new, factor).sum(axis=(1,3))
    
    return reduced


def find_clusters(
    x: np.ndarray,
    y: np.ndarray,
    height = DEFAULT_HEIGHT,
    width = DEFAULT_WIDTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find centroids in event data using heatmap-based initialization and k-means.

    Args:
        x: X coordinates of events
        y: Y coordinates of events
        cfg: Configuration for clustering

    Returns:
        mask: mask with boolean coords that is overlayed on data
    """

    print(x, y)
    heatmap = find_heatmap(x, y, width=width, height=height)
    np.savetxt(str(LOG_DIR / "heatmap.txt"), heatmap, fmt="%d")
    draw_heatmap(heatmap, name="heatmap")

    cx, cy = kmeans_cluster(x, y)
    points = _np.stack([x.astype(_np.float32), y.astype(_np.float32)], axis=1)
