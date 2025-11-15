from dataclasses import dataclass


@dataclass(frozen=False)
class Centroids:
    x_coords: list
    y_coords: list


@dataclass
class CropCoords:
    x1: int
    x2: int
    y1: int
    y2: int
