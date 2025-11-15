import numpy as np
from src.config import HEATMAP_PIXEL_SIZE, CENTROID_EPSILON
from src.logger import get_logger
from src.models import Centroids

logger = get_logger()


def construct_heatmap(frame: np.ndarray, factor: int) -> np.ndarray:
    height, width = frame.shape

    h_new = height // factor
    w_new = width // factor
    reduced = (
        frame[: h_new * factor, : w_new * factor]
        .reshape(h_new, factor, w_new, factor)
        .sum(axis=(1, 3))
    )

    return reduced


def find_furthest_centroid(
    existing_centroids: list[tuple[int, int]], centroids: list[tuple[int, int]]
) -> tuple[int, int]:
    if not existing_centroids:
        # If no existing centroids, return the first candidate
        return centroids[0]

    ex = np.asarray(existing_centroids, dtype=float).reshape(-1, 2)
    cands = np.asarray(centroids, dtype=float)
    mean_distances = (cands[:, None, :] - ex[None, :, :]).sum(axis=2).mean(axis=1)
    chosen = int(mean_distances.argmax())
    return (int(cands[chosen, 0]), int(cands[chosen, 1]))


def get_heatmap_centroids(heatmap: np.ndarray) -> Centroids:
    heatmap_max_val = heatmap.max()
    centroid_inclusion_threshold = heatmap_max_val * CENTROID_EPSILON

    sorted_centroid_indices: list[int] = []
    sorted_contiguous_heatmap = np.argsort(heatmap.ravel())[::-1]
    flat_heatmap = heatmap.ravel()
    for i in range(len(sorted_contiguous_heatmap)):
        idx = sorted_contiguous_heatmap[i]
        if flat_heatmap[idx] > centroid_inclusion_threshold:
            sorted_centroid_indices.append(idx)

    sorted_centroid_indices_xy = [
        (idx % heatmap.shape[1], idx // heatmap.shape[1])
        for idx in sorted_centroid_indices
    ]
    centroids = [sorted_centroid_indices_xy[0]]
    for i, _ in enumerate(sorted_centroid_indices_xy[1:]):
        existing_centroids = sorted_centroid_indices_xy[:i]
        remaining_centroids_xy = sorted_centroid_indices_xy[i + 1 :]
        furthest_centroid = find_furthest_centroid(
            existing_centroids, remaining_centroids_xy
        )
        centroids.append(furthest_centroid)

    return Centroids(
        x_coords=[c[0] for c in centroids], y_coords=[c[1] for c in centroids]
    )


def scale_centroids(centroids: Centroids, scaler: int):
    return Centroids(
        x_coords=centroids.x_coords * scaler, y_coords=centroids.x_coords * scaler
    )


def kmeans(heatmap: np.ndarray, propeller_masks: Centroids) -> np.ndarray:
    return np.array([])


def get_propeller_masks(frame: np.ndarray) -> np.ndarray:
    heatmap = construct_heatmap(frame, factor=HEATMAP_PIXEL_SIZE)
    logger.info(f"Constructed heatmap: {heatmap} of dim {heatmap.shape}")
    centroids = get_heatmap_centroids(heatmap)
    centroids = scale_centroids(centroids, HEATMAP_PIXEL_SIZE)

    logger.info(f"Found centroids: {centroids}")
    return kmeans(heatmap=heatmap, propeller_masks=centroids)


def get_blade_count() -> int:
    return 1
