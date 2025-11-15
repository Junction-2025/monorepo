from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Centroids:
    x_coords: np.ndarray
    y_coords: np.ndarray


@dataclass
class CropCoords:
    x1: int
    x2: int
    y1: int
    y2: int
