import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# Class labels
class_names = ['herb paris', 'karela', 'small weed', 'grass', 'tori', 'horseweed', 'Bhindi', 'weed']

class TRTInference:
    def __init__(self, engine_path):
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for input/output bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            # Get the size in bytes for the data type
            if dtype == trt.DataType.FLOAT:
                dtype_size = 4  # 4 bytes for float32
            elif dtype == trt.DataType.HALF:
                dtype_size = 2  # 2 bytes for float16
            elif dtype == trt.DataType.INT8:
                dtype_size = 1  # 1 byte for int8
            else:
                dtype_size = 4  # Default to 4 bytes
                
            # Calculate total size
            size = trt.volume(shape) * dtype_size
            
            # Get the size in bytes for the data type
            if dtype == trt.DataType.FLOAT:
                dtype_size = 4  # 4 bytes for float32
            elif dtype == trt.DataType.HALF:
                dtype_size = 2  # 2 bytes for float16
            elif dtype == trt.DataType.INT8:
                dtype_size = 1  # 1 byte for int8
            else:
                dtype_size = 4  # Default to 4 bytes
                
            # Calculate total size
            size = trt.volume(shape) * dtype_size
            
            # Allocate CUDA memory
            allocation = cuda.mem_alloc(size)
            
            if is_input:
                self.inputs.append({"index": i, "name": name, "dtype": dtype, "shape": shape, "allocation": allocation})
            else:
                self.outputs.append({"index": i, "name": name, "dtype": dtype, "shape": shape, "allocation": allocation})
            
            self.allocations.append(allocation)
    
    def infer(self, img_input):
        # Copy input data to GPU
        cuda.memcpy_htod(self.inputs[0]["allocation"], img_input.astype(np.float32).ravel())
        
        # Run inference
        self.context.execute_v2(self.allocations)
        
        # Copy output back to CPU
        output_shape = self.outputs[0]["shape"]
        output = np.zeros(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])
        
        return [output]

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
    preds = np.squeeze(predictions[0])  # shape: (N, 85)
    
    # Handle different output formats (some TensorRT engines may have different shapes)
    if len(preds.shape) == 1:
        # If output is flattened, reshape it based on YOLOv5 output format
        num_classes = len(class_names)
        num_boxes = preds.shape[0] // (5 + num_classes)
        preds = preds.reshape(num_boxes, 5 + num_classes)
    
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
    
    # Apply NMS
    if not results:
        return []
    
    boxes_xywh = [r[0] for r in results]
    scores = [r[1] for r in results]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=conf_thres, nms_threshold=iou_thres)
    
    # Handle different return formats of NMSBoxes (OpenCV version differences)
    if len(indices) > 0 and isinstance(indices[0], (list, tuple, np.ndarray)):
        indices = [i[0] for i in indices]
    
    return [results[i] for i in indices]

def main():
    # Initialize TensorRT engine
    trt_model = TRTInference("test.engine")

    # Run inference
    img_num = input("Enter image number: ")
    img_path = f"images/{img_num}.jpg"
    img_input = preprocess(img_path)
    
    # Measure inference time
    start_time = time.time()
    predictions = trt_model.infer(img_input)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"TensorRT Inference Time: {inference_time:.2f} ms")
    
    # Process results
    results = postprocess(predictions)
    
    # Draw results
    original = cv2.imread(img_path)
    original_resized = cv2.resize(original, (640, 640))
    
    for (x, y, w, h), score, cls in results:
        cv2.rectangle(original_resized, (x, y), (x + w, y + h), (0, 0, 0), 2)
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(original_resized, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save and display
    cv2.imwrite("trt_output.jpg", original_resized)
    cv2.imshow("TensorRT Detections", original_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
