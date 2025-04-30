import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load class labels
labelsPath = 'obj.names'
LABELS = open(labelsPath).read().strip().split("\n")

# Load YOLO config and weights
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'

# Colors for bounding boxes
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Detection parameters
confi = 0.5
thresh = 0.5
ln = net.getLayerNames()
layer_indexes = net.getUnconnectedOutLayers().flatten()
ln = [ln[i - 1] for i in layer_indexes]

# Load video file (change to 0 for webcam)
cap = cv2.VideoCapture("crop_weed_video.mp4")

frame_count = 0
max_frames = 30  # Only show this many frames to keep notebook responsive

while True:
    ret, image = cap.read()
    if not ret or frame_count >= max_frames:
        break

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

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

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w - 100, y + h - 100), color, 2)
            print("Predicted -> :", LABELS[classIDs[i]])
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert to RGB and display
    det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(det)
    plt.axis('off')
    plt.title(f"Frame {frame_count + 1}")
    plt.show()

    # Optional: Clear previous frame display (mimics real-time)
    clear_output(wait=True)

    frame_count += 1

cap.release()
print("[STATUS]   : Completed")
