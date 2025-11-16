# exporters.py
import argparse
import shutil
from pathlib import Path
import subprocess
import sys

from ultralytics import YOLO

# Optional Apple Silicon support
try:
    import coremltools as ct
    COREML_OK = True
except:
    COREML_OK = False


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path):
    if src and src.exists():
        shutil.move(str(src), str(dst))


def get_ultra_output(pt_dir: Path, ext: str):
    """
    Ultralytics writes exports next to the .pt file:
        best.pt → best.engine, best.mlpackage
    """
    f = pt_dir / f"best{ext}"
    return f if f.exists() else None


# =========================================================================
# CUSTOM TFLITE CONVERTER (ONNX → SavedModel → TFLite FP16 + SELECT_TF_OPS)
# =========================================================================
TFLITE_CONVERTER_SCRIPT = """
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import sys
import pathlib

onnx_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
saved = out_dir / "saved_model"
tflite_path = out_dir / "model_fp16_selectops.tflite"

# 1. Load ONNX
model = onnx.load(str(onnx_path))
tf_rep = prepare(model)
tf_rep.export_graph(str(saved))

# 2. Convert to TFLite FP16 (SELECT_TF_OPS)
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.target_spec.supported_types = [tf.float16]

tflite = converter.convert()
with open(tflite_path, "wb") as f:
    f.write(tflite)

print("TFLITE_EXPORT_OK")
"""

def run_custom_tflite_export(onnx_file: Path, out_dir: Path):
    mkdir(out_dir)

    script_file = out_dir / "_tmp_tflite_export.py"
    script_file.write_text(TFLITE_CONVERTER_SCRIPT)

    print("[TFLite FP16 SELECT_OPS] Converting ONNX → SavedModel → TFLite…")
    cmd = [sys.executable, str(script_file), str(onnx_file), str(out_dir)]
    subprocess.run(cmd, check=True)

    script_file.unlink(missing_ok=True)
    print("[TFLite FP16 SELECT_OPS] Done.")


# =========================================================================
# MAIN EXPORT PIPELINE
# =========================================================================
def run_all_exports(best_pt: str, export_root: str = "exports", export_apple: bool = False):
    best_pt = Path(best_pt)
    if not best_pt.exists():
        raise FileNotFoundError(f"No such file: {best_pt}")

    export_root = Path(export_root)
    mkdir(export_root)

    TRT_DIR = export_root / "tensorrt"
    TFLITE_DIR = export_root / "tflite"
    APPLE_DIR = export_root / "apple"

    for d in [TRT_DIR, TFLITE_DIR, APPLE_DIR]:
        mkdir(d)

    pt_dir = best_pt.parent

    print(f"\n=== LOADING MODEL: {best_pt} ===\n")
    model = YOLO(str(best_pt))

    # ----------------------------------------------------------
    # 1. TensorRT INT8
    # ----------------------------------------------------------
    # print("[TensorRT INT8] Exporting…")
    # model.export(format="engine", int8=True, device='mps')
    # trt_int8 = get_ultra_output(pt_dir, ".engine")
    # safe_move(trt_int8, TRT_DIR / "model_int8.engine")

    # # ----------------------------------------------------------
    # # 2. ONNX export (needed for TFLite)
    # # ----------------------------------------------------------
    # print("[ONNX] Exporting for TFLite pipeline…")
    # model.export(format="onnx", imgsz=960, opset=17)
    # onnx_file = get_ultra_output(pt_dir, ".onnx")
    # if not onnx_file:
    #     raise RuntimeError("ONNX export did not produce a file.")
    # safe_move(onnx_file, TFLITE_DIR / "model.onnx")
    # onnx_file = TFLITE_DIR / "model.onnx"

    # # ----------------------------------------------------------
    # # 3. Custom TFLite FP16 exporter (SELECT_TF_OPS)
    # # ----------------------------------------------------------
    # try:
    #     run_custom_tflite_export(onnx_file, TFLITE_DIR)
    # except Exception as e:
    #     print("[TFLite FP16 SELECT_OPS] FAILED:", e)

    # ----------------------------------------------------------
    # 4. Optional CoreML
    # ----------------------------------------------------------
    if export_apple:
        if not COREML_OK:
            print("[CoreML] coremltools not installed → skipping.")
        else:
            print("[CoreML FP16] Exporting…")
            model.export(format="coreml", half=True)
            pkg = get_ultra_output(pt_dir, ".mlpackage")
            if pkg:
                safe_move(pkg, APPLE_DIR / "model_fp16.mlpackage")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n=== EXPORT COMPLETE ===\n")
    for f in export_root.rglob("*.*"):
        try:
            print(f"{f} — {f.stat().st_size/1024**2:.2f} MB")
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal YOLO exporter")
    parser.add_argument("best_pt", type=str, help="Path to best.pt")
    parser.add_argument("--out", type=str, default="exports")
    parser.add_argument("--apple", action="store_true")

    args = parser.parse_args()
    run_all_exports(args.best_pt, export_root=args.out, export_apple=args.apple)
