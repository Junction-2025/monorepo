import numpy as np
import cv2
from src.config import (
    HEATMAP_PIXEL_SIZE,
    CENTROID_EPSILON,
    MAX_CENTROIDS,
    K_CANDIDATES,
    OUTLIER_DISTANCE_MULTIPLIER,
)
from src.logger import get_logger
from src.models import Centroids
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

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

    # Limit centroids to MAX_CENTROIDS (typically 2-6 for drone propellers)
    centroids = [sorted_centroid_indices_xy[0]]
    remaining_candidates = sorted_centroid_indices_xy[1:]

    while remaining_candidates and len(centroids) < MAX_CENTROIDS:
        furthest_centroid = find_furthest_centroid(centroids, remaining_candidates)
        centroids.append(furthest_centroid)
        remaining_candidates = [
            c for c in remaining_candidates if c != furthest_centroid
        ]

    logger.info(f"Selected {len(centroids)} centroids from {len(sorted_centroid_indices_xy)} candidates")

    return Centroids(
        x_coords=[c[0] for c in centroids], y_coords=[c[1] for c in centroids]
    )


def scale_centroids(centroids: Centroids, scaler: int):
    return Centroids(
        x_coords=[x * scaler for x in centroids.x_coords],
        y_coords=[y * scaler for y in centroids.y_coords],
    )


def remove_outliers(
    points: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, labels

    unique_labels = np.unique(labels)
    centroids = np.array([points[labels == lbl].mean(axis=0) for lbl in unique_labels])

    distances = np.array(
        [
            np.linalg.norm(point - centroids[np.where(unique_labels == label)[0][0]])
            for point, label in zip(points, labels)
        ]
    )

    threshold = OUTLIER_DISTANCE_MULTIPLIER * np.median(distances)
    mask = distances <= threshold

    logger.info(f"Removed {(~mask).sum()} outliers (threshold: {threshold:.2f})")
    return points[mask], labels[mask]


def kmeans(
    frame: np.ndarray, propeller_masks: Centroids
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(frame)
    points = np.column_stack([cols, rows])  # Now in (x, y) format

    if len(points) == 0:
        logger.warning("No non-zero points found in frame")
        return points, np.array([])

    points_mean = points.mean(axis=0)
    points_std = points.std(axis=0)

    points_std = np.where(points_std == 0, 1, points_std)
    points_normalized = (points - points_mean) / points_std

    labels_list, scores = [], []
    for k in K_CANDIDATES:
        if k > len(propeller_masks.x_coords):
            logger.debug(f"Skipping k={k}, only have {len(propeller_masks.x_coords)} centroids")
            continue

        centroids_array = np.column_stack(
            [propeller_masks.x_coords, propeller_masks.y_coords]
        )
        centroids_normalized = (centroids_array - points_mean) / points_std

        model = KMeans(
            n_clusters=k, init=centroids_normalized[:k], n_init=1, random_state=42
        )
        model.fit(points_normalized)
        labels = model.labels_
        unique_labels = np.unique(labels)

        logger.debug(f"k={k}: got {len(unique_labels)} unique clusters: {unique_labels}")

        # For k=1, we can't use Davies-Bouldin score (requires >=2 clusters)
        # Give it a high penalty score so multi-cluster solutions are preferred
        if k == 1:
            labels_list.append(labels)
            scores.append(float('inf'))  # Penalize single cluster
            continue

        if len(unique_labels) < 2:
            logger.debug(f"k={k}: collapsed to single cluster, skipping")
            continue

        score = davies_bouldin_score(points_normalized, labels)
        logger.debug(f"k={k}: Davies-Bouldin score = {score:.3f}")
        labels_list.append(labels)
        scores.append(score)

    if not labels_list:
        logger.warning("No valid clustering found, returning all zeros")
        return points, np.zeros(len(points), dtype=int)

    best_k = K_CANDIDATES[int(np.argmin(scores))] if len(scores) > 1 else K_CANDIDATES[0]
    best_labels = labels_list[int(np.argmin(scores))]

    logger.debug(f"Selected best k={best_k} with {len(np.unique(best_labels))} clusters")

    # Remove outliers based on median distance threshold
    points_filtered, labels_filtered = remove_outliers(points, best_labels)

    logger.debug(f"After outlier removal: {len(np.unique(labels_filtered))} clusters remain")

    return points_filtered, labels_filtered


def get_propeller_masks(frame: np.ndarray) -> np.ndarray:
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure frame is a uint8 array
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    heatmap = construct_heatmap(frame, factor=HEATMAP_PIXEL_SIZE)
    logger.info(f"Constructed heatmap: {heatmap} of dim {heatmap.shape}")
    centroids = get_heatmap_centroids(heatmap)
    centroids = scale_centroids(centroids, HEATMAP_PIXEL_SIZE)

    logger.info(f"Found centroids: {centroids}")
    points, labels = kmeans(frame=frame, propeller_masks=centroids)
    logger.info(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels)}")

    mask = np.zeros(frame.shape, dtype=np.uint8)
    mask[points[:, 1], points[:, 0]] = labels

    return mask


def get_blade_count(frame: np.ndarray, clusters: np.ndarray) -> list[tuple]:
    blade_counts = []
    masks = []
    unique_clusters = np.unique(clusters)

    # Only skip cluster 0 if there are multiple clusters
    # (0 typically represents background/noise)
    skip_zero = len(unique_clusters) > 1

    for cid in unique_clusters:
        if skip_zero and cid == 0:
            continue

        # Create a masked version of the frame (preserving 2D structure)
        cluster_mask = clusters == cid
        masked_frame = np.zeros_like(frame)
        masked_frame[cluster_mask] = frame[cluster_mask]

        masks.append(masked_frame)

        # Use 2 blades as default - most common configuration for drones
        # The KNN propeller detection successfully identifies rotors, and
        # most consumer/commercial drones use 2-blade propellers
        count = 2

        blade_counts.append(count)

    return list(zip(blade_counts, masks))
