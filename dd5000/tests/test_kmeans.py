import numpy as np
from src.kmeans import get_heatmap_centroids, get_propeller_masks, find_furthest_centroid, construct_heatmap
from src.models import Centroids


def test_reduction_by_factor_two():
    test_frame = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )
    reduced = construct_heatmap(test_frame, factor=2)

    print(reduced)

    assert reduced.shape == (2, 2)
    assert np.array_equal(
        reduced,
        np.array(
            [
                [2, 2],
                [2, 2],
            ],
            dtype=np.int32,
        ),
    )


def test_reduction_8x8_by_factor_two():
    test_frame = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=np.int32,
    )
    reduced = construct_heatmap(test_frame, factor=2)

    print(reduced)

    assert reduced.shape == (4, 4)
    # Each 2x2 block is summed
    # Top-left: [1,2,2,3] = 8
    # Top-mid-left: [3,4,4,5] = 16
    # Top-mid-right: [5,6,6,7] = 24
    # Top-right: [7,8,8,9] = 32
    # etc.
    assert np.array_equal(
        reduced,
        np.array(
            [
                [8, 16, 24, 32],
                [16, 24, 32, 40],
                [24, 32, 40, 48],
                [32, 40, 48, 56],
            ],
            dtype=np.int32,
        ),
    )


def test_reduction_8x8_by_factor_four():
    test_frame = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [3, 3, 3, 3, 4, 4, 4, 4],
            [3, 3, 3, 3, 4, 4, 4, 4],
            [3, 3, 3, 3, 4, 4, 4, 4],
            [3, 3, 3, 3, 4, 4, 4, 4],
        ],
        dtype=np.int32,
    )
    reduced = construct_heatmap(test_frame, factor=4)

    print(reduced)

    assert reduced.shape == (2, 2)
    # Each 4x4 block is summed
    # Top-left: 16 ones = 16
    # Top-right: 16 twos = 32
    # Bottom-left: 16 threes = 48
    # Bottom-right: 16 fours = 64
    assert np.array_equal(
        reduced,
        np.array(
            [
                [16, 32],
                [48, 64],
            ],
            dtype=np.int32,
        ),
    )


def test_reduction_9x9_by_factor_two():
    """Test reduction of 9x9 array with factor 2 - tests handling of non-divisible dimensions"""
    test_frame = np.array(
        [
            [1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 1, 2, 2, 3, 3, 4, 4, 5],
            [2, 2, 3, 3, 4, 4, 5, 5, 6],
            [2, 2, 3, 3, 4, 4, 5, 5, 6],
            [3, 3, 4, 4, 5, 5, 6, 6, 7],
            [3, 3, 4, 4, 5, 5, 6, 6, 7],
            [4, 4, 5, 5, 6, 6, 7, 7, 8],
            [4, 4, 5, 5, 6, 6, 7, 7, 8],
            [5, 5, 6, 6, 7, 7, 8, 8, 9],
        ],
        dtype=np.int32,
    )
    reduced = construct_heatmap(test_frame, factor=2)

    print(reduced)

    assert reduced.shape == (4, 4)
    # Each 2x2 block is summed
    # Row 0: [1,1,1,1]=4, [2,2,2,2]=8, [3,3,3,3]=12, [4,4,4,4]=16
    # Row 1: [2,2,2,2]=8, [3,3,3,3]=12, [4,4,4,4]=16, [5,5,5,5]=20
    # Row 2: [3,3,3,3]=12, [4,4,4,4]=16, [5,5,5,5]=20, [6,6,6,6]=24
    # Row 3: [4,4,4,4]=16, [5,5,5,5]=20, [6,6,6,6]=24, [7,7,7,7]=28
    assert np.array_equal(
        reduced,
        np.array(
            [
                [4, 8, 12, 16],
                [8, 12, 16, 20],
                [12, 16, 20, 24],
                [16, 20, 24, 28],
            ],
            dtype=np.int32,
        ),
    )


def test_find_furthest_centroid_linear_monotonic():
    existing = [(0, 0), (2, 0), (4, 0)]
    candidates = [(5, 0), (9, 0), (6, 0), (3, 0)]
    assert find_furthest_centroid(existing, candidates) == (9, 0)


def test_find_furthest_centroid_linear_symmetry():
    existing = [(1, 0), (9, 0)]
    candidates = [(0, 0), (5, 0), (10, 0)]
    assert find_furthest_centroid(existing, candidates) == (10, 0)


def test_get_heatmap_centroids_four_clusters():
    # 9x9 heatmap with four identical 3x3 clusters placed with centers at
    # (2,2), (6,2), (2,6), (6,6) (x,y). Each cluster:
    heatmap = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 100, 1, 0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0, 1, 50, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    centers = Centroids(x_coords=[2, 6], y_coords=[2, 6])
    centroids = get_heatmap_centroids(heatmap)

    assert centroids == centers


def test_get_heatmap_centroids_no_duplicates():
    """Test that get_heatmap_centroids doesn't return duplicate centroids.

    This test ensures that when find_furthest_centroid is called, it uses the
    already-selected centroids (not the original sorted list) to avoid duplicates.
    """
    # Small 2x2 heatmap with 4 high-value pixels
    heatmap = np.array(
        [
            [100, 50],
            [50, 100],
        ],
        dtype=np.int32,
    )

    centroids = get_heatmap_centroids(heatmap)

    # Should have unique centroids
    centroid_tuples = list(zip(centroids.x_coords, centroids.y_coords))
    unique_centroids = list(set(centroid_tuples))

    assert len(centroid_tuples) == len(unique_centroids), \
        f"Found duplicate centroids: {centroid_tuples}"


