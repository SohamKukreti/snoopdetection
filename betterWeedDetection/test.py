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
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output

# Inference
def infer(engine, context, h_input, d_input, h_output, d_output):
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

# Preprocess image
def preprocess(img_path, img_size=640):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Postprocess with NMS
def postprocess(predictions, conf_thres=0.25, iou_thres=0.45):
    preds = np.reshape(predictions, (-1, predictions.shape[-1]))  # shape: (N, 85)

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

# Main
img_path = "images/1.jpg"
engine = load_engine("best.engine")
context = engine.create_execution_context()
h_input, d_input, h_output, d_output = allocate_buffers(engine)

img_input = preprocess(img_path)
np.copyto(h_input, img_input.ravel())

predictions = infer(engine, context, h_input, d_input, h_output, d_output)
results = postprocess(predictions)

# Draw results
original = cv2.imread(img_path)
original_resized = cv2.resize(original, (640, 640))

for (x, y, w, h), score, cls in results:
    cv2.rectangle(original_resized, (x, y), (x + w, y + h), (0, 0, 0), 2)
    label = f"{class_names[cls]}: {score:.2f}"
    cv2.putText(original_resized, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imwrite("output.jpg", original_resized)
cv2.imshow("Detections", original_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
