import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from src.recording import open_dat
from src.config import BASE_HEIGHT, BASE_WIDTH, DATA_DIR
from pyntcloud import PyntCloud


def parse_args():
    parser = argparse.ArgumentParser(description="Convert event camera .dat file to .ply point cloud")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR/ "drone_idle.dat",
        help="Input .dat file path (default: ../data/drone_idle.dat)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default= DATA_DIR/ "events.ply",
        help="Output .ply file path (default: ./events.ply)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=BASE_WIDTH,
        help="Sensor width in pixels (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=BASE_HEIGHT,
        help="Sensor height in pixels (default: 720)"
    )
    parser.add_argument(
        "--use-timestamp-as-z",
        action="store_true",
        default=True,
        help="Use timestamp as Z coordinate (default: True)"
    )
    parser.add_argument(
        "--no-timestamp-as-z",
        dest="use_timestamp_as_z",
        action="store_false",
        help="Set Z coordinate to 0 (2D sheet)"
    )
    parser.add_argument(
        "--normalise-time",
        action="store_true",
        default=True,
        help="Normalise timestamps to 0-1 range (default: True)"
    )
    parser.add_argument(
        "--no-normalise-time",
        dest="normalize_time",
        action="store_false",
        help="Do not normalise timestamps"
    )
    return parser.parse_args()


def load_events_from_dat(input_file: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Event word bit format: [31-28: polarity] [27-14: y] [13-0: x]
    Returns: (x, y, timestamps, polarity) arrays
    """
    print("Reading events...")
    reader = open_dat(input_file, width=width, height=height)

    w32 = reader.event_words.astype(np.uint32)
    pol_array = ((w32 >> 28) & 0xF).astype(np.uint8)
    pol_array = (pol_array > 0).astype(np.uint8)
    y_array = ((w32 >> 14) & 0x3FFF).astype(np.int64)
    x_array = (w32 & 0x3FFF).astype(np.int64)

    xs = []
    ys = []
    ts = []
    pols = []

    for x, y, t, p in zip(x_array, y_array, reader.timestamps, pol_array):
        xs.append(x)
        ys.append(y)
        ts.append(t)
        pols.append(p)

    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)
    pols = np.array(pols)

    print(f"Loaded {len(xs)} events")
    return xs, ys, ts, pols


def calculate_z_coordinates(timestamps: np.ndarray, use_timestamp_as_z: bool, normalize_time: bool) -> np.ndarray:
    if use_timestamp_as_z:
        z = timestamps.astype(np.float64)
        if normalize_time:
            z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    else:
        z = np.zeros_like(timestamps, dtype=np.float64)

    return z


def create_polarity_colors(polarities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive (1) = white, Negative (0) = red"""
    red = np.where(polarities == 1, 255, 255)
    green = np.where(polarities == 1, 255, 0)
    blue = np.where(polarities == 1, 255, 0)

    return red.astype(np.uint8), green.astype(np.uint8), blue.astype(np.uint8)


def build_point_cloud_dataframe(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                                 red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "x": xs,
        "y": ys,
        "z": zs,
        "red": red,
        "green": green,
        "blue": blue
    })


def save_point_cloud(df: pd.DataFrame, output_file: Path) -> None:
    print("Creating PyntCloud object...")
    cloud = PyntCloud(df)

    print(f"Writing to {output_file}...")
    cloud.to_file(str(output_file))

    print("Done.")


def main():
    args = parse_args()

    xs, ys, ts, pols = load_events_from_dat(
        args.input,
        args.width,
        args.height
    )

    zs = calculate_z_coordinates(
        ts,
        args.use_timestamp_as_z,
        args.normalize_time
    )

    red, green, blue = create_polarity_colors(pols)

    df = build_point_cloud_dataframe(xs, ys, zs, red, green, blue)

    save_point_cloud(df, args.output)


if __name__ == "__main__":
    main()