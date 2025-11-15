from pathlib import Path
import shutil
import random

# Configuration
# Root folder that contains sequences named like fred-0, fred-1, ...
FRED_ROOT = Path(".")   # change this if your fred-* directories are elsewhere

# Where to write the YOLO dataset we create
OUT_ROOT = Path("datasets/fred_event")  # will be created if missing
TRAIN_RATIO = 0.8                       # fraction of data used for training

# Helper functions

def collect_event_pairs():
    """
    Walk each fred-* sequence and collect pairs of (image_path, label_path)
    for the event modality.

    Expected structure inside a sequence folder:
      Event/Frames/*.png|jpg|jpeg    <-- images
      Event_YOLO/*.txt               <-- YOLO label files

    We only keep image files that have a matching .txt label.
    """
    pairs = []

    fred_dirs = sorted(
        d for d in FRED_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("fred-")
    )

    if not fred_dirs:
        raise RuntimeError(f"No 'fred-*' folders found under {FRED_ROOT.resolve()}")

    for seq_dir in fred_dirs:
        frames_dir = seq_dir / "Event" / "Frames"
        labels_dir = seq_dir / "Event_YOLO"

        if not frames_dir.is_dir() or not labels_dir.is_dir():
            print(f"[WARN] Skipping {seq_dir.name}: missing Event/Frames or Event_YOLO")
            continue

        # gather image files (common extensions)
        img_paths = []
        img_paths.extend(sorted(frames_dir.glob("*.png")))
        img_paths.extend(sorted(frames_dir.glob("*.jpg")))
        img_paths.extend(sorted(frames_dir.glob("*.jpeg")))

        if not img_paths:
            print(f"[WARN] No event frames found in {frames_dir}")
            continue

        for img_path in img_paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.is_file():
                pairs.append((img_path, label_path))
            else:
                # label missing for this image; skip it silently by default
                # uncomment the print below if you want to log every missing label
                # print(f"[INFO] No label for {img_path}")
                pass

    if not pairs:
        raise RuntimeError("No (image, label) pairs found. Check your structure and names.")

    print(f"Collected {len(pairs)} (image, label) pairs from event frames.")
    return pairs


def discover_classes(label_paths):
    """
    Read all label files to find which class ids appear.
    Return (nc, sorted_class_ids) where nc is number of classes.
    """
    class_ids = set()
    for lp in label_paths:
        with open(lp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])  # first column is the class id
                class_ids.add(cls)

    if not class_ids:
        raise RuntimeError("No class IDs found in labels.")

    max_id = max(class_ids)
    nc = max_id + 1
    print(f"Found class ids: {sorted(class_ids)} -> nc = {nc}")
    return nc, sorted(class_ids)


def make_dirs(root):
    # create the images/labels folders for train/val
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(img_path, lbl_path, split, out_root, idx):
    """
    Copy an image and its label into the dataset tree under the given split.
    We keep the original filename stem to make it easier to trace files back
    to the original dataset.
    """
    img_out_dir = out_root / "images" / split
    lbl_out_dir = out_root / "labels" / split

    stem = img_path.stem
    img_out = img_out_dir / f"{stem}.png"   # using .png as a consistent image ext
    lbl_out = lbl_out_dir / f"{stem}.txt"

    # If you prefer to keep the original extension use:
    # img_out = img_out_dir / img_path.name
    # lbl_out = lbl_out_dir / lbl_path.name

    shutil.copy2(img_path, img_out)
    shutil.copy2(lbl_path, lbl_out)


def write_yaml(out_root, nc):
    """
    Write a minimal YOLO dataset YAML describing this dataset.
    Classes are given generic names class_0, class_1, ...
    """
    yaml_path = out_root / "fred_event.yaml"

    lines = []
    lines.append("path: .")  # dataset YAML paths are relative to the dataset root
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("")
    lines.append(f"nc: {nc}")
    lines.append("names:")

    # Generic class names
    for i in range(nc):
        lines.append(f"  {i}: class_{i}")

    yaml_path.write_text("\n".join(lines))
    print(f"Wrote dataset YAML to: {yaml_path}")


# Main routine

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = collect_event_pairs()

    # Discover number of classes
    label_paths = [lp for (_, lp) in pairs]
    nc, _ = discover_classes(label_paths)

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    # Create dirs
    make_dirs(OUT_ROOT)

    # Copy files
    for idx, (img, lbl) in enumerate(train_pairs):
        copy_pair(img, lbl, "train", OUT_ROOT, idx)
    for idx, (img, lbl) in enumerate(val_pairs):
        copy_pair(img, lbl, "val", OUT_ROOT, idx)

    # YAML
    write_yaml(OUT_ROOT, nc)

    print("DONE. YOLO event dataset created at:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
