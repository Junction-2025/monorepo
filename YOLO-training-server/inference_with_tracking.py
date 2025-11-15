import cv2
import time
from ultralytics import YOLO
from pathlib import Path
import natsort

# CONFIG
MODEL_PATH = "best.pt"
TEST_DIR = Path("val")
OUTPUT_VIDEO = "drone_tracking.mp4"
IMG_SIZE = 640
USE_TRACKER = True
SHOW_WINDOW = False   # <<< IMPORTANT: NO GUI

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if USE_TRACKER:
        model.tracker = "botsort.yaml"

    image_paths = natsort.natsorted(
        [p for p in TEST_DIR.glob("*.png")] +
        [p for p in TEST_DIR.glob("*.jpg")] +
        [p for p in TEST_DIR.glob("*.jpeg")]
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No images in {TEST_DIR}")

    print(f"Found {len(image_paths)} frames.")

    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h)
    )

    prev_time = time.time()
    frame_count = 0
    fps = 0

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        results = model.predict(frame, imgsz=IMG_SIZE, conf=0.3, device='mps', verbose=True)
        annotated = results[0].plot()

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev_time)
            prev_time = now

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        writer.write(annotated)

        # Only show window if GUI mode is enabled
        if SHOW_WINDOW:
            cv2.imshow("Drone Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"\nSaved video: {OUTPUT_VIDEO}\n")

if __name__ == "__main__":
    main()
