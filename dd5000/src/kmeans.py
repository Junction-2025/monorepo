import numpy as np
from src.config import HEATMAP_PIXEL_SIZE, CENTROID_EPSILON
from src.logger import get_logger
from src.models import Centroids, k_means_maker

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
    remaining_candidates = sorted_centroid_indices_xy[1:]

    while remaining_candidates:
        furthest_centroid = find_furthest_centroid(centroids, remaining_candidates)
        centroids.append(furthest_centroid)
        remaining_candidates = [
            c for c in remaining_candidates if c != furthest_centroid
        ]

    return Centroids(
        x_coords=[c[0] for c in centroids], y_coords=[c[1] for c in centroids]
    )


def scale_centroids(centroids: Centroids, scaler: int):
    return Centroids(
        x_coords=[x * scaler for x in centroids.x_coords],
        y_coords=[y * scaler for y in centroids.y_coords],
    )


def kmeans(frame: np.ndarray, propeller_masks: Centroids) -> np.ndarray:
    k_means_model = k_means_maker(propeller_masks)
    # Convert frame pixels to (x, y) coordinate pairs for clustering
    height, width = frame.shape
    y_coords, x_coords = np.meshgrid(range(height), range(width), indexing="ij")
    pixel_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    k_means_model.fit(pixel_coords)
    labels = k_means_model.predict(pixel_coords)
    return labels.reshape(frame.shape)


def get_propeller_masks(frame: np.ndarray) -> np.ndarray:
    heatmap = construct_heatmap(frame, factor=HEATMAP_PIXEL_SIZE)
    logger.info(f"Constructed heatmap: {heatmap} of dim {heatmap.shape}")
    centroids = get_heatmap_centroids(heatmap)
    centroids = scale_centroids(centroids, HEATMAP_PIXEL_SIZE)

    logger.info(f"Found centroids: {centroids}")
    prediction = kmeans(frame=frame, propeller_masks=centroids)
    logger.info(f"Prediction: {prediction}")

    # Scale prediction back to original frame dimensions
    mask = np.repeat(
        np.repeat(prediction, HEATMAP_PIXEL_SIZE, axis=0), HEATMAP_PIXEL_SIZE, axis=1
    )

    # Pad to match original frame size if needed
    h_diff = frame.shape[0] - mask.shape[0]
    w_diff = frame.shape[1] - mask.shape[1]

    if h_diff > 0 or w_diff > 0:
        # Pad with edge values
        mask = np.pad(mask, ((0, h_diff), (0, w_diff)), mode="edge")

    return mask


def get_blade_count() -> int:
    return 1
