from src.config import DATA_DIR, LOG_DIR
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

    z = v["z"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.invert_yaxis()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_file = LOG_DIR / f"{input_file.stem}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
