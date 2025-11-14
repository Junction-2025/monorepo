import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from recording import open_dat
from pyntcloud import PyntCloud


def parse_args():
    parser = argparse.ArgumentParser(description="Convert event camera .dat file to .ply point cloud")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data/drone_idle.dat",
        help="Input .dat file path (default: ../data/drone_idle.dat)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "events.ply",
        help="Output .ply file path (default: ./events.ply)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Sensor width in pixels (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
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
        dest="normalise_time",
        action="store_false",
        help="Do not normalise timestamps"
    )
    return parser.parse_args()


def main():
    # =============================
    # COMMAND LINE ARGUMENTS
    # =============================
    args = parse_args()

    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    USE_TIMESTAMP_AS_Z = args.use_timestamp_as_z
    NORMALISE_TIME = args.normalise_time
    WIDTH = args.width
    HEIGHT = args.height

    # =============================
    # LOAD EVENTS FROM .dat
    # =============================
    reader = open_dat(INPUT_FILE, width=WIDTH, height=HEIGHT)
    print(reader)
    xs = []
    ys = []
    ts = []
    pols = []

    print("Reading events...")
    for e in reader:
        xs.append(e.x)
        ys.append(e.y)
        ts.append(e.ts)
        pols.append(e.p)

    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)
    pols = np.array(pols)

    print(f"Loaded {len(xs)} events")

    # =============================
    # MAP TO POINT CLOUD Z-DIMENSION
    # =============================
    if USE_TIMESTAMP_AS_Z:
        z = ts.astype(np.float64)
        if NORMALISE_TIME:
            z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    else:
        z = np.zeros_like(xs, dtype=np.float64)

    # =============================
    # COLOUR BY POLARITY
    # =============================
    # Positive polarity = white (255,255,255)
    # Negative polarity = red  (255,0,0)
    red   = np.where(pols == 1, 255, 255)
    green = np.where(pols == 1, 255,   0)
    blue  = np.where(pols == 1, 255,   0)

    # =============================
    # BUILD DATAFRAME FOR PYNTCLOUD
    # =============================
    df = pd.DataFrame({
        "x": xs,
        "y": ys,
        "z": z,
        "red": red.astype(np.uint8),
        "green": green.astype(np.uint8),
        "blue": blue.astype(np.uint8)
    })

    print("Creating PyntCloud object...")
    cloud = PyntCloud(df)

    # =============================
    # SAVE AS .ply
    # =============================
    print(f"Writing to {OUTPUT_FILE} ...")
    cloud.to_file(str(OUTPUT_FILE))

    print("Done.")


if __name__ == "__main__":
    main()