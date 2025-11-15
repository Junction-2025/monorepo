from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray
from src.roo.config import (
    EPSILON,
    MAX_ITERATIONS,
    CONVERGENCE_THRESHOLD,
    OUTLIER_THRESHOLD_MULTIPLIER,
    K_CANDIDATES,
    USE_ADAPTIVE_K,
    DEFAULT_K,
)


def initialize_centroids_from_heatmap(
    heatmap: NDArray[np.int64],
    k: int,
    epsilon: float = EPSILON,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Initialize k centroids from heatmap using stream-centroids initialization.

    Strategy: Find highest value grid as first centroid, then select remaining
    centroids by maximizing distance while ensuring grid value > epsilon * h_max.

    Args:
        heatmap: 2D heatmap of accumulated events
        k: Number of centroids to initialize
        epsilon: Threshold factor (default 0.3 from paper)
        seed: Random seed for reproducibility (only used for fallback case)

    Returns:
        Array of shape (k, 2) containing centroid coordinates [y, x]
    """
    if k <= 0:
        return np.empty((0, 2))

    h_max = np.max(heatmap)
    threshold = epsilon * h_max

    valid_mask = heatmap >= threshold
    valid_positions = np.column_stack(np.nonzero(valid_mask))

    if len(valid_positions) == 0:
        h, w = heatmap.shape
        rng = np.random.default_rng(seed)
        return rng.random((k, 2)) * [[h], [w]]

    max_idx = np.argmax(heatmap)
    first_centroid = np.array([max_idx // heatmap.shape[1], max_idx % heatmap.shape[1]])
    centroids = [first_centroid]

    for _ in range(k - 1):
        if len(valid_positions) == 0:
            break

        distances = np.array(
            [
                np.mean([np.linalg.norm(pos - c) for c in centroids])
                for pos in valid_positions
            ]
        )

        next_idx = np.argmax(distances)
        next_centroid = valid_positions[next_idx]
        centroids.append(next_centroid)

        valid_positions = np.delete(valid_positions, next_idx, axis=0)

    return np.array(centroids, dtype=np.float64)


def assign_events_to_clusters(
    events: NDArray[np.float64],
    centroids: NDArray[np.float64],
) -> NDArray[np.int_]:
    """
    Assign each event to nearest centroid using euclidean distance.

    Args:
        events: Array of shape (n, 2) containing event coordinates
        centroids: Array of shape (k, 2) containing centroid coordinates

    Returns:
        Array of shape (n,) containing cluster assignments
    """
    if len(centroids) == 0:
        return np.zeros(len(events), dtype=np.int_)

    distances = np.linalg.norm(
        events[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )

    return np.argmin(distances, axis=1)


def update_centroids(
    events: NDArray[np.float64],
    labels: NDArray[np.int_],
    k: int,
) -> NDArray[np.float64]:
    """
    Update centroid positions as mean of assigned events.

    Args:
        events: Array of shape (n, 2) containing event coordinates
        labels: Array of shape (n,) containing cluster assignments
        k: Number of clusters

    Returns:
        Array of shape (k, 2) containing updated centroid coordinates
    """
    centroids = np.zeros((k, 2))

    for i in range(k):
        cluster_events = events[labels == i]
        if len(cluster_events) > 0:
            centroids[i] = np.mean(cluster_events, axis=0)

    return centroids


def kmeans_clustering(
    events: NDArray[np.float64],
    initial_centroids: NDArray[np.float64],
    max_iterations: int = MAX_ITERATIONS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    Perform K-means clustering on events with given initial centroids.

    Args:
        events: Array of shape (n, 2) containing event coordinates
        initial_centroids: Array of shape (k, 2) containing initial centroids
        max_iterations: Maximum number of iterations
        convergence_threshold: Threshold for centroid movement convergence

    Returns:
        Tuple of (labels, final_centroids)
    """
    if len(events) == 0 or len(initial_centroids) == 0:
        return np.array([]), initial_centroids.copy()

    centroids = initial_centroids.copy()
    k = len(centroids)
    labels = np.zeros(len(events), dtype=np.int_)

    for _ in range(max_iterations):
        labels = assign_events_to_clusters(events, centroids)

        new_centroids = update_centroids(events, labels, k)

        movement = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids

        if movement < convergence_threshold:
            break

    return labels, centroids


def remove_outliers(
    events: NDArray[np.float64],
    labels: NDArray[np.int_],
    centroids: NDArray[np.float64],
    threshold_multiplier: float = OUTLIER_THRESHOLD_MULTIPLIER,
) -> NDArray[np.bool_]:
    """
    Remove outliers based on median distance from centroid.

    For each cluster, calculates median distance Dm and marks events with
    distance > threshold_multiplier × Dm as outliers.

    Args:
        events: Array of shape (n, 2) containing event coordinates
        labels: Array of shape (n,) containing cluster assignments
        centroids: Array of shape (k, 2) containing centroid coordinates
        threshold_multiplier: Multiplier for median distance threshold (default 3.0 from paper)

    Returns:
        Boolean mask of shape (n,) where True indicates inlier, False indicates outlier
    """
    if len(events) == 0:
        return np.array([], dtype=np.bool_)

    inlier_mask = np.ones(len(events), dtype=np.bool_)
    k = len(centroids)

    for cluster_id in range(k):
        # Get events in this cluster
        cluster_mask = labels == cluster_id
        cluster_events = events[cluster_mask]

        if len(cluster_events) == 0:
            continue

        # Calculate distances to centroid
        distances = np.linalg.norm(cluster_events - centroids[cluster_id], axis=1)

        # Calculate median distance (Dm from paper, Equation 8)
        median_distance = np.median(distances)

        # Mark outliers as events with distance > threshold_multiplier × Dm
        threshold = threshold_multiplier * median_distance
        outliers = distances > threshold

        # Update inlier mask
        cluster_indices = np.where(cluster_mask)[0]
        inlier_mask[cluster_indices[outliers]] = False

    return inlier_mask


def calculate_dispersion(
    events: NDArray[np.float64],
    labels: NDArray[np.int_],
    centroid: NDArray[np.float64],
    cluster_id: int,
) -> float:
    """
    Calculate dispersion (average distance) within a cluster.

    Args:
        events: Array of shape (n, 2) containing event coordinates
        labels: Array of shape (n,) containing cluster assignments
        centroid: Centroid coordinates
        cluster_id: Cluster identifier

    Returns:
        Dispersion value
    """
    cluster_events = events[labels == cluster_id]
    if len(cluster_events) == 0:
        return 0.0

    distances = np.linalg.norm(cluster_events - centroid, axis=1)
    return np.mean(distances)


def calculate_separation(
    centroid_i: NDArray[np.float64],
    centroid_j: NDArray[np.float64],
) -> float:
    """
    Calculate separation (distance) between two centroids.

    Args:
        centroid_i: First centroid coordinates
        centroid_j: Second centroid coordinates

    Returns:
        Separation value
    """
    return float(np.linalg.norm(centroid_i - centroid_j))


def calculate_davies_bouldin_index(
    events: NDArray[np.float64],
    labels: NDArray[np.int_],
    centroids: NDArray[np.float64],
) -> float:
    """
    Calculate Davies-Bouldin Index for clustering quality evaluation.

    Lower DBI indicates better clustering (higher inter-cluster distance,
    lower intra-cluster distance).

    Args:
        events: Array of shape (n, 2) containing event coordinates
        labels: Array of shape (n,) containing cluster assignments
        centroids: Array of shape (k, 2) containing centroid coordinates

    Returns:
        Davies-Bouldin Index value
    """
    k = len(centroids)
    if k <= 1:
        # DBI is undefined for k <= 1, return infinity to prevent selection
        return float("inf")

    dispersions = np.array(
        [calculate_dispersion(events, labels, centroids[i], i) for i in range(k)]
    )

    dbi_sum = 0.0
    for i in range(k):
        max_similarity = 0.0

        for j in range(k):
            if i != j:
                separation = calculate_separation(centroids[i], centroids[j])
                if separation > 0:
                    similarity = (dispersions[i] + dispersions[j]) / separation
                    max_similarity = max(max_similarity, similarity)

        dbi_sum += max_similarity

    return dbi_sum / k


def find_optimal_k(
    heatmap: NDArray[np.int64],
    events: NDArray[np.float64],
    k_candidates: List[int],
) -> Tuple[int, float]:
    """
    Find optimal number of clusters using Davies-Bouldin Index.

    Args:
        heatmap: 2D heatmap for centroid initialization
        events: Array of shape (n, 2) containing event coordinates
        k_candidates: List of candidate k values to evaluate

    Returns:
        Tuple of (optimal_k, best_dbi)
    """
    best_k = k_candidates[0]
    best_dbi = float("inf")

    for k in k_candidates:
        initial_centroids = initialize_centroids_from_heatmap(heatmap, k)

        if len(initial_centroids) < k:
            continue

        labels, centroids = kmeans_clustering(events, initial_centroids)

        dbi = calculate_davies_bouldin_index(events, labels, centroids)

        if dbi < best_dbi:
            best_dbi = dbi
            best_k = k

    return best_k, best_dbi


def heatmap_to_weighted_events(
    heatmap: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Convert heatmap to weighted event positions.

    Each pixel value in the heatmap represents the count of events at that location.
    This function expands the heatmap into individual event coordinates, where a pixel
    with value N contributes N events at that location.

    Args:
        heatmap: 2D heatmap of accumulated event counts

    Returns:
        Tuple of (event_positions, pixel_rows, pixel_cols):
        - event_positions: Array of shape (total_events, 2) containing event coordinates [row, col]
        - pixel_rows: Array of shape (total_events,) indicating which pixel row each event came from
        - pixel_cols: Array of shape (total_events,) indicating which pixel col each event came from
    """
    rows, cols = np.nonzero(heatmap)
    counts = heatmap[rows, cols].astype(np.int64)

    # Expand positions by their counts
    event_positions = np.repeat(np.column_stack([rows, cols]), counts, axis=0)

    # Also track which pixel each event came from
    pixel_rows = np.repeat(rows, counts)
    pixel_cols = np.repeat(cols, counts)

    return event_positions.astype(np.float64), pixel_rows, pixel_cols


def events_labels_to_heatmap_labels(
    event_labels: NDArray[np.int_],
    pixel_rows: NDArray[np.int64],
    pixel_cols: NDArray[np.int64],
    heatmap_shape: Tuple[int, int],
    inlier_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.int_]:
    """
    Convert event-level labels back to pixel-level label map.

    For each pixel that has events, assigns the most common label among its events.
    Outlier events are excluded from consideration.

    Args:
        event_labels: Array of shape (n_events,) containing cluster labels for each event
        pixel_rows: Array of shape (n_events,) indicating which pixel row each event came from
        pixel_cols: Array of shape (n_events,) indicating which pixel col each event came from
        heatmap_shape: Shape of the output label map (height, width)
        inlier_mask: Optional boolean mask indicating which events are inliers (not outliers)

    Returns:
        Array of shape heatmap_shape containing cluster labels for each pixel
        Pixels with no inlier events are labeled as -1
    """
    label_map = np.full(heatmap_shape, -1, dtype=np.int_)

    if len(event_labels) == 0:
        return label_map

    # Apply inlier mask if provided
    if inlier_mask is not None:
        event_labels = event_labels[inlier_mask]
        pixel_rows = pixel_rows[inlier_mask]
        pixel_cols = pixel_cols[inlier_mask]

    if len(event_labels) == 0:
        return label_map

    # For each unique pixel position, find the most common label
    unique_pixels = np.unique(np.column_stack([pixel_rows, pixel_cols]), axis=0)

    for pixel in unique_pixels:
        row, col = pixel
        # Find all events from this pixel
        pixel_mask = (pixel_rows == row) & (pixel_cols == col)
        pixel_event_labels = event_labels[pixel_mask]

        if len(pixel_event_labels) == 0:
            continue

        # Assign most common label (mode)
        unique_labels, counts = np.unique(pixel_event_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]

        label_map[row, col] = most_common_label

    return label_map


def locate_centroids(
    scene: NDArray[np.int64],
    k: int = DEFAULT_K,
    remove_outliers_flag: bool = True,
    outlier_threshold: float = OUTLIER_THRESHOLD_MULTIPLIER,
    use_adaptive_k: bool = USE_ADAPTIVE_K,
    k_candidates: List[int] | None = None,
) -> Tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """
    Locate centroids in heatmap using stream-centroids initialization and K-means.

    Args:
        scene: 2D heatmap array where values represent intensity/count
        k: Number of centroids to locate (default from config, used when adaptive selection disabled)
        remove_outliers_flag: Whether to apply outlier removal (default True)
        outlier_threshold: Threshold multiplier for outlier detection (default 3.0 from paper)
        use_adaptive_k: Enable adaptive K selection using Davies-Bouldin Index (default from config)
        k_candidates: List of K values to evaluate for adaptive selection (default from config)

    Returns:
        Tuple of (label_map, centers_x, centers_y):
        - label_map: Array of shape scene.shape with cluster labels for each pixel (-1 for empty/outlier pixels)
        - centers_x: Array of centroid x-coordinates
        - centers_y: Array of centroid y-coordinates
    """
    # Convert heatmap to weighted events, tracking which pixel each event came from
    event_positions, pixel_rows, pixel_cols = heatmap_to_weighted_events(scene)

    if len(event_positions) == 0:
        # No events, return empty label map and centroids
        return np.full(scene.shape, -1, dtype=np.int_), np.array([]), np.array([])

    # Determine optimal K using adaptive selection if enabled
    if use_adaptive_k:
        if k_candidates is None:
            k_candidates = K_CANDIDATES
        k, best_dbi = find_optimal_k(scene, event_positions, k_candidates)
        print(f"Adaptive K selection: k={k}, DBI={best_dbi:.4f}")
    else:
        print(f"Using fixed k={k}")

    # Initialize centroids from heatmap
    initial_centroids = initialize_centroids_from_heatmap(scene, k)

    # Perform K-means clustering on weighted events
    event_labels, final_centroids = kmeans_clustering(
        event_positions, initial_centroids
    )

    # Remove outliers if requested
    inlier_mask = None
    if remove_outliers_flag:
        inlier_mask = remove_outliers(
            event_positions,
            event_labels,
            final_centroids,
            threshold_multiplier=outlier_threshold,
        )

    # Convert event-level labels back to pixel-level label map
    # Outliers will be excluded (marked as -1)
    label_map = events_labels_to_heatmap_labels(
        event_labels, pixel_rows, pixel_cols, scene.shape, inlier_mask
    )

    # Return as (label_map, x, y) coordinates (note: heatmap uses row, col indexing)
    return label_map, final_centroids[:, 1].copy(), final_centroids[:, 0].copy()
