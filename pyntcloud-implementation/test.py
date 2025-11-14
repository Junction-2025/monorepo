import numpy as np
import pandas as pd
from evio.src.evio.source.dat_file import open_dat
from pyntcloud import PyntCloud

# =============================
# CONFIG
# =============================
INPUT_FILE = "/Users/harshitpoudel/Desktop/JUNCTION/evio/fan_const_rpm.dat"
OUTPUT_FILE = "events.ply"
USE_TIMESTAMP_AS_Z = True    # If False → z = 0 (2-D sheet)
NORMALISE_TIME = True        # Scale timestamps to 0–1 range for visibility

WIDTH = 1280
HEIGHT = 720

# =============================
# LOAD EVENTS FROM .dat
# =============================
reader = open_dat(INPUT_FILE, width=WIDTH, height=HEIGHT)

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
cloud.to_file(OUTPUT_FILE)

print("Done.")
