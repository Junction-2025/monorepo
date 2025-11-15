from ultralytics import YOLO
from pathlib import Path
import sys

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
    validate_dataset()

    model = YOLO(BASE_MODEL)

    print("Starting HIGH-SPEED training...\n")

    model.train(
        data=str(YAML_PATH),
        imgsz=IMG_SIZE,
        epochs=50,
        batch=BATCH_SIZE,
        workers=8,

        # Performance options
        amp=True,           # use mixed precision (FP16) for speed
        pretrained=True,
        cos_lr=True,        # cosine learning rate schedule
        device=DEVICE,
        cache="ram",        # load all data into RAM (if you have enough memory)

        # Logging options
        project="runs/fred_event",
        name="yolo11s_fast_train",
    )

    # Run validation after training
    print("\nRunning validation...\n")
    model.val(data=str(YAML_PATH), device=DEVICE)

    # Export the trained model to TensorRT (FP16) for fast inference
    print("\nExporting TensorRT FP16 engine for ultra-fast inference...\n")
    model.export(format="engine", half=True, device=DEVICE)

    print("\n✓ Done. Ready for high-speed TensorRT inference.\n")


if __name__ == "__main__":
    main()
