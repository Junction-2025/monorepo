import numpy as np
from src.utils import draw_png

def find_centroids(frame: np.ndarray) -> np.ndarray:
    # Ensure shape is HxWx3 and dtype is uint8, then save as PNG ("frame.png")
    draw_png(frame)
    return frame