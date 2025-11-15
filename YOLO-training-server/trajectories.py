import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import natsort
import csv

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best-rgb.pt"   # or .engine
TEST_DIR = Path("val-rgb")
OUTPUT_VIDEO = "drone_tracking_full_rgb.mp4"
OUTPUT_PLOT = "drone_trajectories_rgb.png"
OUTPUT_CSV = "tracks_rgb.csv"

IMG_SIZE = 960        # MUST match your TensorRT engine (960x960)
USE_TRACKER = True
DEVICE = "mps"        # use "mps" on M1/M2/M3, or 0 for GPU (A30 server)

# ============================================================
# UTIL
# ============================================================

def get_color(tid):
    """Generate deterministic color for each track ID."""
    return (
        int((tid * 37) % 255),
        int((tid * 19) % 255),
        int((tid * 73) % 255),
    )

# ============================================================
# MAIN
# ============================================================

def main():
    # ------------------ Load Model ------------------
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if USE_TRACKER:
        model.tracker = "botsort.yaml"

    # ------------------ Load Frames ------------------
    image_paths = natsort.natsorted(
        list(TEST_DIR.glob("*.png")) +
        list(TEST_DIR.glob("*.jpg")) +
        list(TEST_DIR.glob("*.jpeg"))
    )
    if not image_paths:
        raise RuntimeError(f"No frames found in {TEST_DIR}")

    print(f"Found {len(image_paths)} frames.\n")

    # Frame size for video writer
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h)
    )

    # trajectories[track_id] = [(cx, cy), ...]
    trajectories = {}

    # Save all tracking data to CSV
    csv_file = open(OUTPUT_CSV, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "cx", "cy"])

    # ============================================================
    # PROCESS FRAMES
    # ============================================================

    frame_idx = 0
    total_ms = 0

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))

        # ------------------ Inference ------------------
        t0 = time.time()
        results = model.track(
            frame,
            imgsz=IMG_SIZE,
            conf=0.3,
            device=DEVICE,
            verbose=False
        )
        infer_ms = (time.time() - t0) * 1000
        total_ms += infer_ms
        frame_idx += 1

        # ------------------ Parse Results ------------------
        annotated = results[0].plot()

        # Each detection with tracking ID
        if results[0].boxes is not None:
            ids = results[0].boxes.id
            xyxy = results[0].boxes.xyxy

            if ids is not None:
                for tid, box in zip(ids, xyxy):
                    tid = int(tid)
                    x1, y1, x2, y2 = box.tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Store trajectory
                    trajectories.setdefault(tid, []).append((cx, cy))

                    # Save CSV
                    csv_writer.writerow([frame_idx, tid, cx, cy])

                    # Draw trail
                    pts = trajectories[tid]
                    for i in range(1, len(pts)):
                        cv2.line(
                            annotated,
                            (int(pts[i-1][0]), int(pts[i-1][1])),
                            (int(pts[i][0]), int(pts[i][1])),
                            get_color(tid),
                            2
                        )

        # Write time stats
        fps = 1000 / infer_ms
        cv2.putText(annotated, f"{infer_ms:.2f} ms  ({fps:.1f} FPS)",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

        # Write to video
        writer.write(annotated)

        print(f"[{frame_idx}/{len(image_paths)}]  {infer_ms:.2f} ms ({fps:.1f} FPS)")

    writer.release()
    csv_file.close()

    avg_ms = total_ms / frame_idx
    print("\n========================================")
    print(f"Saved full tracking video: {OUTPUT_VIDEO}")
    print(f"Saved tracking CSV:        {OUTPUT_CSV}")
    print(f"Avg latency:               {avg_ms:.2f} ms ({1000/avg_ms:.1f} FPS)")
    print("========================================\n")

    # ============================================================
    # GENERATE TRAJECTORY PLOT
    # ============================================================

    plt.figure(figsize=(10, 8))
    plt.title("Drone Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")

    for tid, pts in trajectories.items():
        pts = np.array(pts)
        plt.plot(pts[:,0], pts[:,1], label=f"ID {tid}", linewidth=2)
        plt.scatter(pts[:,0], pts[:,1], s=5)

    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.savefig(OUTPUT_PLOT, dpi=250)
    plt.close()

    print(f"Saved trajectory plot: {OUTPUT_PLOT}\n")

# ============================================================
if __name__ == "__main__":
    main()
