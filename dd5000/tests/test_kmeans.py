from src.kmeans import construct_heatmap
import numpy as np


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
