from ultralytics import YOLO
from pathlib import Path
import sys
import argparse

# Configuration
# Path to the base YOLO model to use for training
BASE_MODEL = "yolo11s.pt"        # good balance of speed and accuracy
IMG_SIZE = 960                   # input image size
BATCH_SIZE = 32                  # adjust based on your GPU memory
DEVICE = 0                       # which GPU to use (0 = first GPU)

# Paths for dataset and config
DATASET_DIR = Path(__file__).parent / "datasets" / "fred_event"
YAML_PATH = DATASET_DIR / "fred_event.yaml"

# --- Validation helpers ---
def assert_exists(path, msg):
    if not path.exists():
        print(f"\nERROR: {msg} at: {path}\n")
        sys.exit(1)

def validate_dataset():
    # Check that all required dataset files and folders exist
    assert_exists(YAML_PATH, "Dataset YAML missing")
    for f in ["images/train", "images/val", "labels/train", "labels/val"]:
        assert_exists(DATASET_DIR / f, f"Missing dataset folder: {f}")
    print("Dataset OK ✓\n")

# --- Main training routine ---
def main():
    parser = argparse.ArgumentParser(description="Train FRED YOLO model with flexible options")
    parser.add_argument('--base_model', type=str, default=BASE_MODEL, help='Base YOLO model to use for training')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Input image size')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--device', type=str, default=str(DEVICE), help='Device for training (cpu, cuda, mps, 0, 1, etc)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--dataset_dir', type=str, default=str(DATASET_DIR), help='Dataset directory')
    parser.add_argument('--yaml', type=str, default=str(YAML_PATH), help='Path to dataset YAML')
    parser.add_argument('--project', type=str, default='runs/fred_event', help='Project directory for logs/weights')
    parser.add_argument('--name', type=str, default='yolo11s_fast_train', help='Run name')
    parser.add_argument('--no_export', action='store_true', help='Skip TensorRT export step')
    args = parser.parse_args()

    global DATASET_DIR, YAML_PATH
    DATASET_DIR = Path(args.dataset_dir)
    YAML_PATH = Path(args.yaml)

    validate_dataset()

    model = YOLO(args.base_model)
    print("Starting HIGH-SPEED training...\n")
    model.train(
        data=str(YAML_PATH),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=8,
        amp=True,
        pretrained=True,
        cos_lr=True,
        device=args.device,
        cache="ram",
        project=args.project,
        name=args.name,
    )
    print("\nRunning validation...\n")
    model.val(data=str(YAML_PATH), device=args.device)
    if not args.no_export:
        print("\nExporting TensorRT FP16 engine for ultra-fast inference...\n")
        model.export(format="engine", half=True, device=args.device)
    print("\n✓ Done. Ready for high-speed TensorRT inference.\n")

if __name__ == "__main__":
    main()
