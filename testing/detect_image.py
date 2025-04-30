import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Set image number manually (avoid input())
image_num = 1  # Change this to any number from 1 to 10
image_path = f'images/image_{image_num}.jpeg'

# Check if the selected image exists
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found.")
else:
    # Load class labels YOLO model was trained on
    labelsPath = 'obj.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # Load YOLO model weights and config
    weightsPath = 'crop_weed_detection.weights'
    configPath = 'crop_weed.cfg'
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    print("[INFO]     : loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Load input image
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # Parameters
    confi = 0.5
    thresh = 0.5

    # Get output layer names
    ln = net.getLayerNames()
    layer_indexes = net.getUnconnectedOutLayers().flatten()
    ln = [ln[i - 1] for i in layer_indexes]

    # Create blob and do forward pass
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO]     : YOLO took {:.6f} seconds".format(end - start))

    # Collect detections
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confi:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-max suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
    print("[INFO]     : Detections done, drawing bounding boxes...")

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w - 100, y + h - 100), color, 2)
            print("[OUTPUT]   : detected label ->", LABELS[classIDs[i]])
            print("[ACCURACY] : accuracy ->", confidences[i])
            text = "{} : {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert to RGB for matplotlib display
    det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show output inline in notebook
    plt.figure(figsize=(12, 8))
    plt.imshow(det)
    plt.axis('off')
    plt.title('YOLO Detection Output')
    plt.show()

    print("[STATUS]   : Completed")
    print("[END]")
