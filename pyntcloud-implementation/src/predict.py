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
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Whether to output png to logs or not (default: False)",
    )

    return parser.parse_args()


def visualize(x, y, z, input_file: Path) -> None:
    fig = plt.figure()
    # downsample by factor of 10
    x_ds = x[::10]
    y_ds = y[::10]
    z_ds = z[::10]

    # make room for a 2D projection (left) and the 3D view (right)
    fig.set_size_inches(12, 6)
    ax2d = fig.add_subplot(1, 2, 1)
    sc = ax2d.scatter(x_ds, y_ds, c=z_ds, s=1, cmap="viridis")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_title("XY projection (colored by z)")
    ax2d.invert_yaxis()
    plt.colorbar(sc, ax=ax2d, label="z")

    # Close the initial composite figure and create separate figures for 2D and 3D
    plt.close(fig)

    # 2D output
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    sc2 = ax2.scatter(x_ds, y_ds, c=z_ds, s=1, cmap="viridis")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("XY projection (colored by z)")
    ax2.invert_yaxis()
    plt.colorbar(sc2, ax=ax2, label="z")
    output_file_2d = LOG_DIR / f"{input_file.stem}_xy.png"
    fig2.savefig(output_file_2d, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3D output
    fig3 = plt.figure(figsize=(6, 6))
    ax3d = fig3.add_subplot(1, 1, 1, projection="3d")
    ax3d.scatter(x_ds, y_ds, z_ds, s=1)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.invert_yaxis()
    output_file_3d = LOG_DIR / f"{input_file.stem}_3d.png"
    fig3.savefig(output_file_3d, dpi=300, bbox_inches="tight")
    plt.close(fig3)


def main():
    args = parse_args()
    input_file = args.input

    ply = PlyData.read(input_file)
    v = ply["vertex"].data
    x, y = v["x"], v["y"]
    z = v["z"]

    if args.visualize:
        visualize(x, y, z, input_file)


if __name__ == "__main__":
    main()
