#!/bin/zsh

# Paths
DATASET_DIR="datasets/fred_event"
YAML_PATH="$DATASET_DIR/fred_event.yaml"
PROJECT="runs/fred_event"

# Model list
for MODEL in yolov12n.pt yolov12s.pt yolov12m.pt yolov12l.pt; do
  NAME="${MODEL%.pt}_event"
  echo "Training $MODEL on event camera data..."
  python train_FRED_model.py \
    --base_model $MODEL \
    --imgsz 960 \
    --batch 32 \
    --device 0 \
    --epochs 50 \
    --dataset_dir $DATASET_DIR \
    --yaml $YAML_PATH \
    --project $PROJECT \
    --name $NAME
done