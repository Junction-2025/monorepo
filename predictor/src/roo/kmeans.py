from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans

K_MEANS = KMeans(n_clusters=4, random_state=42, n_init=10)


def locate_centroids(
    scene: np.ndarray,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run KMeans on heatmap and return cluster center coordinates.

    Args:
        scene: 2D heatmap array where values represent intensity/count
        random_state: random seed for reproducibility

    Returns:
        Tuple of two 1D numpy arrays: (centers_x, centers_y)
    """


    K_MEANS.fit(scene)
    centers = K_MEANS.cluster_centers_
    return centers[:, 0].copy(), centers[:, 1].copy()
