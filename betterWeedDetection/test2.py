import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# Load class labels
class_names = ['herb paris', 'karela', 'small weed', 'grass', 'tori', 'horseweed', 'Bhindi', 'weed']

# Load TensorRT engine
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Allocate buffers for inference
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

# Preprocessing function
def preprocess_image(img_path, input_shape=(640, 640)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_transposed = np.transpose(img_rgb, (2, 0, 1))  # CHW
    img_input = np.expand_dims(img_transposed, axis=0)  # NCHW
    return img_input.astype(np.float32), img

# Postprocessing with NMS
def postprocess(predictions, conf_thres=0.25, iou_thres=0.45):
    preds = np.squeeze(predictions).reshape(-1, 85)  # YOLOv5 output format

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

    if not results:
        return []

    boxes_xywh = [r[0] for r in results]
    scores = [r[1] for r in results]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=conf_thres, nms_threshold=iou_thres)
    indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
    return [results[i] for i in indices]

# Run inference
def run_inference(engine_path, img_path):
    engine = load_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    img_input, original_img = preprocess_image(img_path)
    np.copyto(inputs[0][0], img_input.ravel())

    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    results = postprocess(outputs[0][0])

    original_resized = cv2.resize(original_img, (640, 640))
    for (x, y, w, h), score, cls in results:
        cv2.rectangle(original_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(original_resized, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("output.jpg", original_resized)
    cv2.imshow("Detections", original_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function
run_inference("best.engine", "images/1.jpg")
