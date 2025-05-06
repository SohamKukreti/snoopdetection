import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os

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
            
            # Allocate CUDA memory
            allocation = cuda.mem_alloc(size)
            
            if is_input:
                self.inputs.append({"index": i, "name": name, "dtype": dtype, "shape": shape, "allocation": allocation})
            else:
                self.outputs.append({"index": i, "name": name, "dtype": dtype, "shape": shape, "allocation": allocation})
            
            self.allocations.append(allocation)
    
    def infer(self, img_input):
        try:
            # Copy input data to GPU
            cuda.memcpy_htod(self.inputs[0]["allocation"], img_input.astype(np.float32).ravel())
            
            # Run inference
            try:
                # Try the newer API first
                self.context.execute_v2(self.allocations)
            except AttributeError:
                # Fall back to older API if execute_v2 is not available
                print("Falling back to execute_async...")
                self.context.execute_async(batch_size=1, bindings=self.allocations, stream_handle=cuda.Stream().handle)
            
            # Copy output back to CPU
            output_shape = self.outputs[0]["shape"]
            output = np.zeros(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])
            
            return [output]
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            # Return an empty array with the expected shape
            return [np.zeros(self.outputs[0]["shape"], dtype=np.float32)]

# Preprocess image
def preprocess(img, img_size=640):
    if img is None:
        print("Error: Invalid image")
        return None
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    img_batched = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
    return img_batched

# Postprocess with NMS
def postprocess(predictions, conf_thres=0.25, iou_thres=0.45):
    try:
        preds = np.squeeze(predictions[0])  # shape: (N, 85)
        
        # Handle different output formats (some TensorRT engines may have different shapes)
        if len(preds.shape) == 1:
            # If output is flattened, reshape it based on YOLOv5 output format
            num_classes = len(class_names)
            num_boxes = preds.shape[0] // (5 + num_classes)
            preds = preds.reshape(num_boxes, 5 + num_classes)
        
        # Ensure we have the correct number of columns
        if preds.shape[1] < 5 + len(class_names):
            print(f"Warning: Output shape {preds.shape} doesn't match expected format")
            return []
        
        boxes = preds[:, :4]
        objectness = preds[:, 4]
        class_probs = preds[:, 5:5+len(class_names)]  # Only take as many classes as we have names for
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
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        import traceback
        traceback.print_exc()
        return []

def draw_results(image, results):
    """Draw bounding boxes and labels on the image"""
    for (x, y, w, h), score, cls in results:
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{class_names[cls]}: {score:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw filled rectangle for text background
        cv2.rectangle(
            image, 
            (x, y - text_height - 5), 
            (x + text_width, y), 
            (0, 255, 0), 
            -1
        )
        
        # Draw text
        cv2.putText(
            image, 
            label, 
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            2
        )
    
    return image

def main():
    try:
        # Initialize TensorRT engine
        print("Loading TensorRT engine...")
        trt_model = TRTInference("test.engine")
        print("TensorRT engine loaded successfully")
        
        # Create window for display
        window_name = "TensorRT Object Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Set initial image number
        img_num = 1000
        max_img_num = 2500
        
        # Display info
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Loop through all images
        running = True
        while running and img_num <= max_img_num:
            try:
                # Construct image path
                img_path = f"1000data/weed{img_num}.jpg"
                
                # Check if file exists
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}, trying next...")
                    img_num += 1
                    continue
                
                # Read image
                print(f"Processing image: {img_path}")
                frame = cv2.imread(img_path)
                
                if frame is None:
                    print(f"Could not read image: {img_path}")
                    img_num += 1
                    continue
                
                # Original image size for display
                display_frame = cv2.resize(frame, (640, 640))
                
                # Preprocess
                img_input = preprocess(frame)
                if img_input is None:
                    img_num += 1
                    continue
                
                # Inference
                inference_start = time.time()
                predictions = trt_model.infer(img_input)
                inference_time = (time.time() - inference_start) * 1000  # ms
                
                # Process results
                results = postprocess(predictions)
                
                # Draw results on the frame
                result_frame = draw_results(display_frame.copy(), results)
                
                # Calculate and display FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1:  # Update FPS every second
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Add info text
                info_text = f"Image: {img_num} | FPS: {fps:.1f} | Inference: {inference_time:.1f}ms | Detected: {len(results)}"
                cv2.putText(result_frame, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow(window_name, result_frame)
                
                # Wait for key press to control flow
                key = cv2.waitKey(100)  # Delay between frames (100ms = 10 FPS simulation)
                
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    print("User requested exit")
                    running = False
                    break
                elif key == 32:  # Spacebar to pause/resume
                    print("Paused - press any key to continue")
                    cv2.waitKey(0)
                
                # Save output image if needed (optional)
                # cv2.imwrite(f"output/trt_output_{img_num}.jpg", result_frame)
                
                # Move to next image
                img_num += 1
                
            except Exception as e:
                print(f"Error processing image {img_num}: {e}")
                import traceback
                traceback.print_exc()
                img_num += 1  # Continue with next image
        
        cv2.destroyAllWindows()
        print("Finished processing all images")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
