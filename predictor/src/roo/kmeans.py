from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray

# Constants from paper
EPSILON = 0.3  # Threshold factor for centroid selection
MAX_ITERATIONS = 100  # Maximum K-means iterations
CONVERGENCE_THRESHOLD = 1e-4  # Centroid movement threshold for convergence


def initialize_centroids_from_heatmap(
    heatmap: NDArray[np.float64],
    k: int,
    epsilon: float = EPSILON,
) -> NDArray[np.float64]:
    """
    Initialize k centroids from heatmap using stream-centroids initialization.

    Strategy: Find highest value grid as first centroid, then select remaining
    centroids by maximizing distance while ensuring grid value > epsilon * h_max.

    Args:
        heatmap: 2D heatmap of accumulated events
        k: Number of centroids to initialize
        epsilon: Threshold factor (default 0.3 from paper)

    Returns:
        Array of shape (k, 2) containing centroid coordinates [y, x]
    """
    if k <= 0:
        return np.empty((0, 2))

    # Find maximum value in heatmap
    h_max = np.max(heatmap)
    threshold = epsilon * h_max

    # Get all grid positions with values above threshold
    valid_mask = heatmap >= threshold
    valid_positions = np.column_stack(np.nonzero(valid_mask))

    if len(valid_positions) == 0:
        # Fallback: use random positions if no valid grids
        h, w = heatmap.shape
        rng = np.random.default_rng()
        return rng.random((k, 2)) * [[h], [w]]

    # First centroid: grid with highest value
    max_idx = np.argmax(heatmap)
    first_centroid = np.array([max_idx // heatmap.shape[1], max_idx % heatmap.shape[1]])
    centroids = [first_centroid]

    # Select remaining centroids by maximizing average distance
    for _ in range(k - 1):
        if len(valid_positions) == 0:
            break

        # Calculate distances from each valid position to all current centroids
        distances = np.array([
            np.mean([np.linalg.norm(pos - c) for c in centroids])
            for pos in valid_positions
        ])

        # Select position with maximum average distance
        next_idx = np.argmax(distances)
        next_centroid = valid_positions[next_idx]
        centroids.append(next_centroid)

        # Remove selected position from valid positions
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

    # Calculate distances from each event to each centroid
    distances = np.linalg.norm(
        events[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )

    # Assign to nearest centroid
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
        # Assign events to clusters
        labels = assign_events_to_clusters(events, centroids)

        # Update centroids
        new_centroids = update_centroids(events, labels, k)

        # Check convergence
        movement = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids

        if movement < convergence_threshold:
            break

    return labels, centroids


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
        return 0.0

    # Calculate dispersion for each cluster
    dispersions = np.array([
        calculate_dispersion(events, labels, centroids[i], i)
        for i in range(k)
    ])

    # Calculate maximum similarity for each cluster
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
    heatmap: NDArray[np.float64],
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
    best_dbi = float('inf')

    for k in k_candidates:
        # Initialize centroids
        initial_centroids = initialize_centroids_from_heatmap(heatmap, k)

        if len(initial_centroids) < k:
            continue

        # Perform clustering
        labels, centroids = kmeans_clustering(events, initial_centroids)

        # Calculate DBI
        dbi = calculate_davies_bouldin_index(events, labels, centroids)

        # Update best k if this is better
        if dbi < best_dbi:
            best_dbi = dbi
            best_k = k

    return best_k, best_dbi


def heatmap_to_weighted_events(
    heatmap: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert heatmap to weighted event positions.

    Each pixel value in the heatmap represents the count of events at that location.
    This function expands the heatmap into individual event coordinates, where a pixel
    with value N contributes N events at that location.

    Args:
        heatmap: 2D heatmap of accumulated event counts

    Returns:
        Array of shape (total_events, 2) containing event coordinates [row, col]
    """
    # Get non-zero positions and their counts
    rows, cols = np.nonzero(heatmap)
    counts = heatmap[rows, cols].astype(np.int64)

    # Expand positions by their counts (each count becomes multiple event points)
    event_positions = np.repeat(
        np.column_stack([rows, cols]),
        counts,
        axis=0
    )

    return event_positions.astype(np.float64)


def locate_centroids(
    scene: NDArray[np.float64],
    k: int = 4,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Locate centroids in heatmap using stream-centroids initialization and K-means.

    Args:
        scene: 2D heatmap array where values represent intensity/count
        k: Number of centroids to locate (default 4)

    Returns:
        Tuple of two 1D numpy arrays: (centers_x, centers_y)
    """
    # Convert heatmap to weighted event positions
    # Each pixel value represents the number of events at that location
    event_positions = heatmap_to_weighted_events(scene)

    if len(event_positions) == 0:
        # No events, return empty centroids
        return np.array([]), np.array([])

    # Initialize centroids from heatmap (using original heatmap for initialization)
    initial_centroids = initialize_centroids_from_heatmap(scene, k)

    # Perform K-means clustering on weighted events
    _, final_centroids = kmeans_clustering(event_positions, initial_centroids)

    # Return as (x, y) coordinates (note: heatmap uses row, col indexing)
    return final_centroids[:, 1].copy(), final_centroids[:, 0].copy()
