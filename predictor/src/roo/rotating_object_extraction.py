import numpy as np

from src.config import HEATMAP_PIXEL_SIZE
from src.yolo.yolo import CropCoords
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
