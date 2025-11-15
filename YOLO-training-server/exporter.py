# exporters.py
import os
from pathlib import Path
from ultralytics import YOLO

# Optional imports
try:
    import onnx
    from onnxsim import simplify
    ONNX_OK = True
except:
    ONNX_OK = False

try:
    import coremltools as ct
    COREML_OK = True
except:
    COREML_OK = False


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Main modular export function
# ---------------------------------------------------------
def run_all_exports(best_pt_path: str, export_root: str = "exports"):
    """
    Run all GPU, CPU, Apple Silicon, and edge-device exports.
    Can be safely called from any script after training finishes.
    """
    best_pt_path = Path(best_pt_path)
    if not best_pt_path.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt_path}")

    export_root = Path(export_root)
    mkdir(export_root)

    GPU_DIR = export_root / "gpu"
    CPU_DIR = export_root / "cpu"
    APPLE_DIR = export_root / "apple"
    EDGE_DIR = export_root / "edge"
    BASE_DIR = export_root / "base"

    for d in [GPU_DIR, CPU_DIR, APPLE_DIR, EDGE_DIR, BASE_DIR]:
        mkdir(d)

    print("\n=== LOADING MODEL ===")
    model = YOLO(str(best_pt_path))

    # -------------------------------------------
    # 1. GPU EXPORTS
    # -------------------------------------------
    print("\n[GPU] Exporting TensorRT FP16…")
    model.export(
        format="engine",
        half=True,
        device=0,
        imgsz=960,
        path=str(GPU_DIR / "model_fp16.engine")
    )

    print("[GPU] Exporting TensorRT INT8…")
    model.export(
        format="engine",
        int8=True,
        device=0,
        imgsz=960,
        path=str(GPU_DIR / "model_int8.engine")
    )

    # -------------------------------------------
    # 2. BASE EXPORTS
    # -------------------------------------------
    print("\n[BASE] Exporting ONNX…")
    onnx_out = BASE_DIR / "model.onnx"
    model.export(format="onnx", opset=12, simplify=False, path=str(onnx_out))

    if ONNX_OK:
        print("[BASE] Simplifying ONNX…")
        onnx_model = onnx.load(str(onnx_out))
        simplified, _ = simplify(onnx_model)
        onnx.save(simplified, str(BASE_DIR / "model_simplified.onnx"))

    print("[BASE] Exporting TorchScript…")
    model.export(format="torchscript", path=str(BASE_DIR / "model.torchscript"))

    # -------------------------------------------
    # 3. CPU EXPORTS
    # -------------------------------------------
    print("\n[CPU] Exporting OpenVINO FP16…")
    model.export(format="openvino", half=True, path=str(CPU_DIR / "openvino_fp16"))

    # INT8 stub
    (CPU_DIR / "INT8_README.txt").write_text(
        "Run OpenVINO POT tool to quantize to INT8:\n"
        "pot -c pot_config.json -m openvino_fp16.xml -w openvino_fp16.bin\n"
    )

    # -------------------------------------------
    # 4. APPLE EXPORTS
    # -------------------------------------------
    if COREML_OK:
        print("\n[APPLE] Exporting CoreML FP16…")
        fp16_path = APPLE_DIR / "model_fp16.mlpackage"
        model.export(format="coreml", half=True, path=str(fp16_path))

        print("[APPLE] Quantizing CoreML → INT8…")
        mlmodel = ct.models.MLModel(str(fp16_path))
        mlmodel_int8 = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=8
        )
        mlmodel_int8.save(str(APPLE_DIR / "model_int8.mlpackage"))
    else:
        print("\n[APPLE] CoreMLTools missing; skipping CoreML.")

    # -------------------------------------------
    # 5. EDGE EXPORTS
    # -------------------------------------------
    print("\n[EDGE] Exporting TFLite FP16…")
    model.export(format="tflite", half=True, path=str(EDGE_DIR / "model_fp16.tflite"))

    print("[EDGE] Exporting TFLite INT8…")
    model.export(format="tflite", int8=True, path=str(EDGE_DIR / "model_int8.tflite"))

    (EDGE_DIR / "compile_edgetpu.sh").write_text(
        "edgetpu_compiler model_int8.tflite\n"
    )

    # -------------------------------------------
    # Summary
    # -------------------------------------------
    print("\n=== SIZE SUMMARY ===")
    for f in export_root.rglob("*.*"):
        try:
            size = f.stat().st_size / 1024**2
            print(f"{f}: {size:.2f} MB")
        except:
            pass

    print("\n=== ALL EXPORTS COMPLETE ===\n")
