import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np

def load_tensorrt_engine(engine_path):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context

def infer_tensorrt(engine, context, image):
    # This is a placeholder. Actual preprocessing and inference will depend on your model.
    # You must adapt this to your model's input/output format.
    raise NotImplementedError("TensorRT inference needs to be implemented for your model.")

def load_tflite_model(model_path):
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print("tflite_runtime failed or Flex ops required, falling back to TensorFlow's TFLite interpreter...")
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

def infer_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    # Only print once per run for clarity
    if not hasattr(infer_tflite, "_printed_shape"):
        print(f"TFLite model expects input shape: {input_shape}")
        infer_tflite._printed_shape = True
    # Handle both NHWC and NCHW
    if len(input_shape) == 4:
        if input_shape[1] > 10:  # NHWC: [1, height, width, 3]
            h, w = input_shape[1], input_shape[2]
            img = cv2.resize(image, (w, h))
            img = np.expand_dims(img, axis=0).astype(np.float32)
        else:  # NCHW: [1, 3, height, width]
            h, w = input_shape[2], input_shape[3]
            img = cv2.resize(image, (w, h))
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = np.expand_dims(img, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")
    # Normalize if model expects it (uncomment if needed)
    # img = img / 255.0
    # Only print once per run for clarity
    if not hasattr(infer_tflite, "_printed_img_shape"):
        print("Input image shape to interpreter:", img.shape)
        infer_tflite._printed_img_shape = True
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def load_onnx_model(model_path):
    import onnxruntime as ort
    session = ort.InferenceSession(model_path)
    return session

def infer_onnx(session, image):
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img = cv2.resize(image, (input_shape[2], input_shape[3]))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    outputs = session.run(None, {input_name: img})
    return outputs[0]

def load_tf_savedmodel(model_dir):
    import tensorflow as tf
    model = tf.saved_model.load(model_dir)
    return model

def infer_tf_savedmodel(model, image):
    import tensorflow as tf
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (640, 640))
    img = tf.expand_dims(img, 0)
    outputs = model(img)
    return outputs

def load_coreml_model(model_path):
    import coremltools as ct
    model = ct.models.MLModel(model_path)
    return model

def infer_coreml(model, image):
    import numpy as np
    from PIL import Image
    spec = model.get_spec()
    input_desc = spec.description.input[0]
    input_name = input_desc.name
    # Handle image input type
    if hasattr(input_desc.type, 'imageType'):
        # Get expected size
        h = input_desc.type.imageType.height
        w = input_desc.type.imageType.width
        # Convert OpenCV image (BGR) to PIL (RGB)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((w, h))
        input_dict = {input_name: pil_img}
        output = model.predict(input_dict)
        return output
    # Handle multiArrayType as before
    elif hasattr(input_desc.type, 'multiArrayType'):
        shape = [d for d in input_desc.type.multiArrayType.shape]
        if len(shape) == 4:
            if shape[1] == 3:  # (1, 3, H, W)
                h, w = shape[2], shape[3]
                img = cv2.resize(image, (w, h))
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0).astype(np.float32)
            elif shape[3] == 3:  # (1, H, W, 3)
                h, w = shape[1], shape[2]
                img = cv2.resize(image, (w, h))
                img = np.expand_dims(img, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unsupported CoreML input shape: {shape}")
            input_dict = {input_name: img}
            output = model.predict(input_dict)
            return output
        else:
            raise ValueError(f"Unsupported CoreML input shape: {shape}")
    else:
        raise ValueError("Unsupported CoreML input type: neither imageType nor multiArrayType")

def main():
    parser = argparse.ArgumentParser(description="Multi-backend inference script")
    parser.add_argument('--model', type=str, required=True, help='Path to model file or directory')
    parser.add_argument('--backend', type=str, required=True, choices=['tensorrt', 'tflite', 'onnx', 'savedmodel', 'coreml'], help='Model backend type')
    parser.add_argument('--data', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--show', action='store_true', help='Show window (GUI)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    image_paths = sorted([str(p) for p in Path(args.data).glob('*.png')] + [str(p) for p in Path(args.data).glob('*.jpg')] + [str(p) for p in Path(args.data).glob('*.jpeg')])
    if not image_paths:
        print(f"No images found in {args.data}")
        return

    if args.backend == 'tensorrt':
        engine, context = load_tensorrt_engine(args.model)
    elif args.backend == 'tflite':
        interpreter = load_tflite_model(args.model)
    elif args.backend == 'onnx':
        session = load_onnx_model(args.model)
    elif args.backend == 'savedmodel':
        model = load_tf_savedmodel(args.model)
    elif args.backend == 'coreml':
        model = load_coreml_model(args.model)
    else:
        raise ValueError('Unsupported backend')

    inference_times = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        t0 = time.time()
        if args.backend == 'tensorrt':
            output = infer_tensorrt(engine, context, img)
        elif args.backend == 'tflite':
            output = infer_tflite(interpreter, img)
        elif args.backend == 'onnx':
            output = infer_onnx(session, img)
        elif args.backend == 'savedmodel':
            output = infer_tf_savedmodel(model, img)
        elif args.backend == 'coreml':
            output = infer_coreml(model, img)
        else:
            output = None
        t1 = time.time() - t0
        inference_times.append(t1)
        # Save or visualize output as needed (placeholder)
        out_img_path = os.path.join(args.output, os.path.basename(img_path))
        cv2.imwrite(out_img_path, img)  # Replace with annotated image if available

        if args.show:
            cv2.imshow("Inference Output", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    avg_inf = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_inf:.4f} seconds/frame")
    with open(os.path.join(args.output, "metrics.txt"), "w") as f:
        f.write(f"Average inference time: {avg_inf:.4f} seconds/frame\n")
        f.write(f"Total frames: {len(image_paths)}\n")
        f.write(f"FPS (approx): {1/avg_inf:.2f}\n")

if __name__ == "__main__":
    main()
