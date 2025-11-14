"""Simple YOLOv8 import and basic usage example."""

import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO


def predict_image(model: YOLO, source: str | Path | np.ndarray, save_path: str | None = None):
    """
    Run prediction on an image or numpy array.
    
    Args:
        model: YOLO model instance
        source: Image path, URL, or numpy array (BGR format)
        save_path: Optional path to save annotated results
    """
    print(f"\nRunning prediction on: {source}")
    results = model.predict(source)
    
    # Process results
    for result in results:
        # Print detection info
        print(f"\nDetections: {len(result.boxes)} objects found")
        
        # Show results (displays image with bounding boxes)
        result.show()
        
        # Save results with annotations
        if save_path:
            result.save(save_path)
            print(f"Results saved to {save_path}")
        else:
            result.save("output.jpg")
            print("Results saved to output.jpg")
        
        # Print class names and confidence scores
        if len(result.boxes) > 0:
            print("\nDetected objects:")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  - {class_name}: {conf:.2%} confidence")
    
    return results


def main():
    """Test YOLOv8 import and create a model."""
    parser = argparse.ArgumentParser(description="YOLOv8 prediction test")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (optional, defaults to test image if available)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotated image (default: output.jpg)",
    )
    args = parser.parse_args()
    
    print("Importing YOLOv8...")
    
    # Load a pretrained YOLOv8 model
    # Options: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 
    #          'yolov8m.pt' (medium), 'yolov8l.pt' (large), 'yolov8x.pt' (xlarge)
    model = YOLO(args.model)
    
    print(f"Model loaded: {model.model_name}")
    print("YOLOv8 import successful!")
    
    # Run prediction
    if args.image:
        predict_image(model, args.image, args.output)
    else:
        # Example: predict on a test image or numpy array
        print("\nNo image provided. Use --image <path> to specify an image.")
        print("Example: python test_yolo.py --image path/to/image.jpg")
    
    return model


if __name__ == "__main__":
    main()

