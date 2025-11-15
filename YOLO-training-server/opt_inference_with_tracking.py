import cv2
import time
from ultralytics import YOLO
from pathlib import Path
import natsort
import argparse

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
    parser = argparse.ArgumentParser(description="YOLO TensorRT/ONNX Inference with Tracking Pipeline")
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to YOLO model or engine')
    parser.add_argument('--data', type=str, default=str(TEST_DIR), help='Directory with input images')
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO, help='Output video filename')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Inference image size')
    parser.add_argument('--device', type=str, default='0', help='Device for inference (cpu, cuda, mps, 0, 1, etc)')
    parser.add_argument('--tracker', type=str, default='botsort.yaml', help='Tracker config file')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show window (GUI)')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    if USE_TRACKER:
        model.tracker = args.tracker

    image_paths = natsort.natsorted(
        list(Path(args.data).glob("*.png")) +
        list(Path(args.data).glob("*.jpg")) +
        list(Path(args.data).glob("*.jpeg"))
    )
    if len(image_paths) == 0:
        raise RuntimeError(f"No image frames found in: {args.data}")
    print(f"Found {len(image_paths)} frames.")

    first_frame = cv2.imread(str(image_paths[0]))
    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h)
    )
    total_ms = 0
    total_frames = 0
    inference_times = []
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.time()
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False
        )
        infer_ms = (time.time() - t0) * 1000
        inference_times.append(infer_ms)
        total_ms += infer_ms
        total_frames += 1
        fps = 1000 / infer_ms
        annotated = results[0].plot()
        cv2.putText(
            annotated, f"{infer_ms:.2f} ms", (12, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2
        )
        cv2.putText(
            annotated, f"{fps:.1f} FPS", (12, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2
        )
        writer.write(annotated)
        print(f"[{total_frames}]  {infer_ms:.2f} ms   ({fps:.1f} FPS)")
        if args.show:
            cv2.imshow("TensorRT Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    avg_ms = total_ms / total_frames if total_frames > 0 else 0
    print("\n=========================================")
    print(f"Total frames:      {total_frames}")
    print(f"Average latency:   {avg_ms:.2f} ms")
    print(f"Average FPS:       {1000/avg_ms:.1f}" if avg_ms > 0 else "Average FPS: N/A")
    print(f"Saved video:       {args.output}")
    print("=========================================\n")
    with open(args.output + "_metrics.txt", "w") as f:
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Average latency: {avg_ms:.2f} ms\n")
        f.write(f"Average FPS: {1000/avg_ms:.1f}\n" if avg_ms > 0 else "Average FPS: N/A\n")

if __name__ == "__main__":
    main()
