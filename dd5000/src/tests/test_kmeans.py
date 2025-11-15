from src.kmeans import construct_heatmap
import numpy as np

def test_reduction_by_factor_four():
    frame = np.empty((2, 0), dtype=np.int32)
    reduced = construct_heatmap(frame, factor=4)
    assert reduced.shape == (100 // 4, 100 // 4)

