"""
Simplified AOI (Area of Interest) detection with outlier removal.

Functional approach to detecting rotating objects using:
- Heatmap-based initialization
- K-means clustering
- Distance-based outlier removal
"""

import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray


class AOI(NamedTuple):
    """Area of Interest representing a detected rotating object."""

    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: NDArray[np.bool_]


def build_heatmap(
    x: NDArray[np.int64],
    y: NDArray[np.int64],
    width: int,
    height: int,
    bin_size: int = 4,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """
    Build spatial heatmap from event coordinates.

    Returns:
        heat: Binned event counts [grid_h, grid_w]
        xs: X grid coordinates [grid_h, grid_w]
        ys: Y grid coordinates [grid_h, grid_w]
    """
    gw = width // bin_size + 1
    gh = height // bin_size + 1
    heat = np.zeros((gh, gw), dtype=np.int32)

    bin_x = np.clip(x // bin_size, 0, gw - 1)
    bin_y = np.clip(y // bin_size, 0, gh - 1)

    # Vectorized accumulation using np.add.at
    flat_indices = bin_y * gw + bin_x
    np.add.at(heat.ravel(), flat_indices, 1)

    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    return heat, xs, ys


def init_centroids_from_heatmap(
    heat: NDArray[np.int32],
    xs: NDArray[np.int32],
    ys: NDArray[np.int32],
    k: int,
    bin_size: int = 4,
) -> NDArray[np.float32]:
    """
    Initialize k centroids from heatmap using farthest-point sampling.

    Strategy:
    1. Start with hottest bin
    2. Choose subsequent centroids far from existing ones, weighted by heat

    Returns:
        centroids: [k, 2] array of (x, y) pixel coordinates
    """
    flat_idx = np.argsort(heat.ravel())[::-1]
    coords = np.column_stack((xs.ravel()[flat_idx], ys.ravel()[flat_idx]))
    vals = heat.ravel()[flat_idx]

    valid = vals > 0
    coords = coords[valid]
    vals = vals[valid]

    if len(coords) == 0:
        raise RuntimeError("Empty heatmap, no events")

    chosen = [coords[0]]

    for _ in range(1, k):
        dists = np.min(
            np.linalg.norm(coords[:, None, :] - np.array(chosen)[None, :, :], axis=-1),
            axis=1,
        )
        scores = dists * (vals / (vals.max() + 1e-9))
        idx = np.argmax(scores)
        chosen.append(coords[idx])

    # Convert bin indices to pixel coordinates
    centroids_pix = np.array(chosen, dtype=np.float32)
    centroids_pix = (centroids_pix + 0.5) * bin_size
    return centroids_pix


def kmeans(
    points: NDArray[np.float32],
    init_centroids: NDArray[np.float32],
    max_iters: int = 20,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Simple k-means clustering.

    Returns:
        labels: [N] cluster assignments
        centroids: [k, 2] final centroid positions
    """
    centroids = init_centroids.copy()
    k = centroids.shape[0]
    labels = np.zeros(points.shape[0], dtype=np.int32)

    for _ in range(max_iters):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = np.argmin(dists, axis=1)

        if np.all(new_labels == labels):
            break

        labels = new_labels
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = points[mask].mean(axis=0)

    return labels, centroids


def remove_outliers(
    points: NDArray[np.float32],
    centroid: NDArray[np.float32],
    factor: float = 3.0,
) -> NDArray[np.bool_]:
    """
    Remove outliers based on distance from centroid.

    Points with distance > factor * median_distance are considered outliers.

    Returns:
        inlier_mask: Boolean mask of inliers
    """
    dists = np.linalg.norm(points - centroid[None, :], axis=1)
    median_dist = np.median(dists)
    return dists <= (factor * median_dist)


def detect_aois(
    x: NDArray[np.int64],
    y: NDArray[np.int64],
    width: int,
    height: int,
    num_clusters: int,
    bin_size: int = 4,
    outlier_factor: float = 3.0,
    max_kmeans_iters: int = 20,
) -> list[AOI]:
    """
    Detect Areas of Interest (rotating objects) in event data.

    Process:
    1. Build heatmap from events
    2. Initialize centroids from hottest regions
    3. Run k-means on event coordinates
    4. Remove outliers per cluster
    5. Compute bounding boxes

    Args:
        x, y: Event coordinates
        width, height: Sensor dimensions
        num_clusters: Number of rotating objects to detect
        bin_size: Heatmap bin size in pixels
        outlier_factor: Outlier threshold multiplier
        max_kmeans_iters: Maximum k-means iterations

    Returns:
        List of AOI objects with centroids, bboxes, and masks
    """
    # Build heatmap
    heat, xs, ys = build_heatmap(x, y, width, height, bin_size)

    # Initialize centroids
    centroids_init = init_centroids_from_heatmap(heat, xs, ys, num_clusters, bin_size)

    # Run k-means
    points = np.stack([x.astype(np.float32), y.astype(np.float32)], axis=1)
    labels, _ = kmeans(points, centroids_init, max_kmeans_iters)

    # Extract AOIs per cluster
    aoi_list = []
    for k in range(num_clusters):
        mask_k = labels == k
        if not np.any(mask_k):
            continue

        pts_k = points[mask_k]
        c = pts_k.mean(axis=0)

        # Remove outliers
        inlier_mask_local = remove_outliers(pts_k, c, outlier_factor)
        if not np.any(inlier_mask_local):
            continue

        # Map back to global mask
        mask_indices = np.nonzero(mask_k)[0]
        inlier_indices = mask_indices[inlier_mask_local]
        global_mask = np.zeros_like(x, dtype=bool)
        global_mask[inlier_indices] = True

        # Compute bounding box with padding
        x_in = x[global_mask]
        y_in = y[global_mask]
        x1 = int(max(0, x_in.min() - 5))
        y1 = int(max(0, y_in.min() - 5))
        x2 = int(min(width, x_in.max() + 5))
        y2 = int(min(height, y_in.max() + 5))

        aoi = AOI(
            centroid=(float(c[0]), float(c[1])),
            bbox=(x1, y1, x2, y2),
            mask=global_mask,
        )
        aoi_list.append(aoi)

    return aoi_list
