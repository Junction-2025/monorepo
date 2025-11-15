from src.kmeans import construct_heatmap
import numpy as np


def test_reduction_by_factor_four():
    # Create a frame with 100x100 grid of points
    # Frame format: [x_coords, y_coords] where each has 100 points
    x_coords = np.arange(100, dtype=np.int32)
    y_coords = np.arange(100, dtype=np.int32)
    frame = np.array([x_coords, y_coords], dtype=np.int32)

    reduced = construct_heatmap(frame, factor=4)
    assert reduced.shape == (100 // 4, 100 // 4)

