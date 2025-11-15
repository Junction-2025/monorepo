from dataclasses import dataclass
from sklearn.cluster import KMeans
import numpy as np


@dataclass(frozen=False)
class Centroids:
    x_coords: list
    y_coords: list


@dataclass
class CropCoords:
    x1: int
    x2: int
    y1: int
    y2: int


def k_means_maker(centroids: Centroids):
    init_centers = np.column_stack([centroids.x_coords, centroids.y_coords])
    return KMeans(n_clusters=2, init=init_centers, n_init=1)
