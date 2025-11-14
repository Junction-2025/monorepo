from src.config import DATA_DIR
from pathlib import Path
import argparse

from plyfile import PlyData
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get k-means clusters from rotors")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "events.ply",
        help="Input .ply file path (default: ../data/drone_idle.ply)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input

    ply = PlyData.read(input_file)
    v = ply["vertex"].data
    x, y = v["x"], v["y"]

    plt.scatter(x, y, s=1)
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()
