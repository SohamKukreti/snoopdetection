import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Class labels
class_names = ['herb paris', 'karela', 'small weed', 'grass', 'tori', 'horseweed', 'Bhindi', 'weed']

# Load TensorRT engine
def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
        print("Loaded TensorRT engine.")
        return engine

# Allocate host/device buffers
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    print(f"Allocated input size: {h_input.shape}, output size: {h_output.shape}")
    return h_input, d_input, h_output, d_output

# Inference execution
def infer(engine, context, h_input, d_input, h_output, d_output):
    stream = cuda.Stream()
    bindings = [int(d_input), int(d_output)]

    # Copy input to device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Set binding shape for dynamic input
    context.set_binding_shape(0, (1, 3, 640, 640))

    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy output back
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

# Preprocess image
def preprocess(img_path, img_size=640):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_transposed, axis=0)
    print("Preprocessed image shape:", img_input.shape)
    return img_input, img_resized

# Postprocess
def postprocess(predictions, conf_thres=0.25, iou_thres=0.45):
    preds = predictions.reshape(-1, 85)  # (N, 85)
    print("Postprocess: total predictions from model:", preds.shape)

    boxes = preds[:, :4]
    objectness = preds[:, 4]
    class_probs = preds[:, 5:]

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_ids)), class_ids]
    scores = objectness * class_scores

    results = []
    for box, score, cls in zip(boxes, scores, class_ids):
        if score > conf_thres:
            cx, cy, w, h = box
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            results.append(([x, y, int(w), int(h)], float(score), int(cls)))

    print("Detections before NMS:", len(results))
    if not results:
        return []

    boxes_xywh = [r[0] for r in results]
    scores = [r[1] for r in results]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)

    indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
    print("Detections after NMS:", len(indices))
    return [results[i] for i in indices]

# Main
def main():
    img_path = "images/1.jpg"
    engine = load_engine("best.engine")
    context = engine.create_execution_context()

    h_input, d_input, h_output, d_output = allocate_buffers(engine)

    # Load and preprocess input image
    img_input, img_display = preprocess(img_path)
    np.copyto(h_input, img_input.ravel())

    # Run inference
    predictions = infer(engine, context, h_input, d_input, h_output, d_output)

    print("Raw output shape:", predictions.shape)
    print("First few output values:", predictions[:10])
    print("Max value in output:", np.max(predictions))

    # Reshape predictions for postprocessing
    predictions = predictions.reshape(1, -1, 85)
    results = postprocess(predictions)

    # Draw detections
    for (x, y, w, h), score, cls in results:
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 0, 0), 2)
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(img_display, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite("output.jpg", img_display)
    cv2.imshow("Detections", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
