import cv2
import time
from ultralytics import YOLO
from pathlib import Path
import natsort
import argparse

# CONFIG
MODEL_PATH = "best-custom.pt"
TEST_DIR = Path("/Users/harshitpoudel/Desktop/JUNCTION/monorepo/evio/output-frames")
OUTPUT_VIDEO = "drone_tracking.mp4"
IMG_SIZE = 640
USE_TRACKER = True

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference with Tracking Pipeline")
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to YOLO model or engine')
    parser.add_argument('--data', type=str, default=str(TEST_DIR), help='Directory with input images')
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO, help='Output video filename')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Inference image size')
    parser.add_argument('--device', type=str, default='mps', help='Device for inference (cpu, cuda, mps, 0, 1, etc)')
    parser.add_argument('--tracker', type=str, default='botsort.yaml', help='Tracker config file')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show window (GUI)')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    if USE_TRACKER:
        model.tracker = args.tracker

    image_paths = natsort.natsorted(
        [p for p in Path(args.data).glob("*.png")] +
        [p for p in Path(args.data).glob("*.jpg")] +
        [p for p in Path(args.data).glob("*.jpeg")]
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No images in {args.data}")

    print(f"Found {len(image_paths)} frames.")

    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h)
    )

    prev_time = time.time()
    frame_count = 0
    fps = 0
    inference_times = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        t0 = time.time()
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=True)
        infer_time = time.time() - t0
        inference_times.append(infer_time)
        annotated = results[0].plot()

        # Draw all bounding boxes with confidence and class
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
            label = f"{cls} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev_time)
            prev_time = now

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        writer.write(annotated)

        if args.show:
            cv2.imshow("Drone Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_inf = sum(inference_times) / len(inference_times)
    print(f"\nSaved video: {args.output}\n")
    print(f"Average inference time: {avg_inf:.4f} seconds/frame")
    with open(args.output + "_metrics.txt", "w") as f:
        f.write(f"Average inference time: {avg_inf:.4f} seconds/frame\n")
        f.write(f"Total frames: {len(image_paths)}\n")
        f.write(f"FPS (approx): {1/avg_inf:.2f}\n")

if __name__ == "__main__":
    main()
