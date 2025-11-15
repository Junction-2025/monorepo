import cv2
import time
from ultralytics import YOLO
from pathlib import Path
import natsort

# ============================================================
# CONFIG
# ============================================================

# Path to your TensorRT FP16 engine
MODEL_PATH = "runs/fred_event/yolo11s_fast_train/weights/best.engine"

# Test frames directory
TEST_DIR = Path("datasets/fred_event/images/val")

# Output video path
OUTPUT_VIDEO = "drone_tracking_trt.mp4"

# FIXED inference size (must match TensorRT engine build)
IMG_SIZE = 960                # IMPORTANT: engine was built at 960x960

# Tracker
USE_TRACKER = True

# Headless (no cv2.imshow)
SHOW_WINDOW = False


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading TensorRT engine: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if USE_TRACKER:
        model.tracker = "botsort.yaml"

    # Load test frames
    image_paths = natsort.natsorted(
        list(TEST_DIR.glob("*.png")) +
        list(TEST_DIR.glob("*.jpg")) +
        list(TEST_DIR.glob("*.jpeg"))
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No image frames found in: {TEST_DIR}")

    print(f"Found {len(image_paths)} frames.")

    # Get frame dimensions
    first_frame = cv2.imread(str(image_paths[0]))
    h, w = first_frame.shape[:2]

    # Output video writer
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h)
    )

    total_ms = 0
    total_frames = 0

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # ------------------ TensorRT Inference + Timing ------------------
        t0 = time.time()
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,        # MUST be 960 for TensorRT engine
            conf=0.3,
            device=0,
            verbose=False
        )
        infer_ms = (time.time() - t0) * 1000
        total_ms += infer_ms
        total_frames += 1

        fps = 1000 / infer_ms

        # ------------------ Overlay stats ------------------
        annotated = results[0].plot()

        cv2.putText(
            annotated, f"{infer_ms:.2f} ms", (12, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2
        )

        cv2.putText(
            annotated, f"{fps:.1f} FPS", (12, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2
        )

        # Save frame to video
        writer.write(annotated)

        # Print to terminal
        print(f"[{total_frames}]  {infer_ms:.2f} ms   ({fps:.1f} FPS)")

        # Optional GUI (disabled for SSH)
        if SHOW_WINDOW:
            cv2.imshow("TensorRT Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    # Summary
    avg_ms = total_ms / total_frames

    print("\n=========================================")
    print(f"Total frames:      {total_frames}")
    print(f"Average latency:   {avg_ms:.2f} ms")
    print(f"Average FPS:       {1000/avg_ms:.1f}")
    print(f"Saved video:       {OUTPUT_VIDEO}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
