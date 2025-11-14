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

    if scene.size == 0:
        return np.array([]), np.array([])

    # Convert heatmap to coordinate list weighted by intensity
    # Find all non-zero positions in the heatmap
    y_indices, x_indices = np.nonzero(scene)

    if len(x_indices) == 0:
        # No non-zero values in heatmap
        return np.array([]), np.array([])

    # Get the intensity values at these positions
    intensities = scene[y_indices, x_indices]

    coords_list = []
    for x, y, intensity in zip(x_indices, y_indices, intensities):
        repeat_count = min(int(intensity), 100)  # cap to avoid memory issues
        for _ in range(max(1, repeat_count)):
            coords_list.append([x, y])

    coords = np.array(coords_list, dtype=np.float32)

    print(
        f"Clustering {len(coords)} points from heatmap with {len(x_indices)} non-zero cells"
    )

    K_MEANS.fit(coords)
    centers = K_MEANS.cluster_centers_
    return centers[:, 0].copy(), centers[:, 1].copy()
