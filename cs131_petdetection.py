import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Configuration
MAX_PLAYROOM_CAPACITY = 3
IMAGE_PATH = "pet_daycare3.jpg"
ALERT_TEXT = ""

# Define dynamic zone ratios to cover the ENTIRE image
ZONE_RATIOS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

print("Loading Optimized Deep Learning Models...")
device = "mps" if torch.backends.mps.is_available() else 'cpu'
#device = "cuda" if torch.cuda.is_available() else "cpu"
detector_model = YOLO("yolov8n.pt").to(device)
classifier_model = YOLO("yolov8m-cls.pt").to(device)

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"CRITICAL ERROR: Could not find or load image at '{IMAGE_PATH}'")
    exit()

frame_h, frame_w = frame.shape[:2]
focus_zone_coords = (ZONE_RATIOS * [frame_w, frame_h]).astype(np.int32)
ZONES = {"focus_zone": focus_zone_coords}

# Run detector with half-precision (if it detects a CUDA compatible GPU)
results = detector_model(frame, half=(device == 'cuda'))
zone_count = 0

# Data structures to handle batch classification
dog_crops = []
dog_indices = []
detection_list = []

# Draw the zone boundary
cv2.polylines(frame, [ZONES["focus_zone"]], True, (0, 255, 255), 2)

# Stage 1: Collect detections and identify dogs for batching
for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = detector_model.names[class_id]

        if class_name in ["dog", "person"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bottom_center = (int((x1 + x2) / 2), y2)

            if cv2.pointPolygonTest(ZONES["focus_zone"], bottom_center, False) >= 0:
                zone_count += 1
                cv2.circle(frame, bottom_center, 6, (0, 0, 255), -1)

                detection_data = {"coords": (x1, y1, x2, y2), "class": class_name, "label": class_name.title()}

                if class_name == "dog":
                    pad = 10
                    crop = frame[max(0, y1-pad):min(frame_h, y2+pad), max(0, x1-pad):min(frame_w, x2+pad)]
                    if crop.size > 0:
                        dog_crops.append(crop)
                        dog_indices.append(len(detection_list))

                detection_list.append(detection_data)

# Stage 2: Perform Batched Classification
if dog_crops:
    cls_results = classifier_model(dog_crops, verbose=False)
    for i, res in enumerate(cls_results):
        prob = res.probs.top1conf.item()
        idx_in_list = dog_indices[i]
        if prob > 0.4:
            breed_name = res.names[res.probs.top1].replace("_", " ").title()
            detection_list[idx_in_list]["label"] = f"Dog: {breed_name} ({prob:.2f})"
        else:
            detection_list[idx_in_list]["label"] = "Dog: Breed Unclear"

# Stage 3: Print the results
for det in detection_list:
    x1, y1, x2, y2 = det["coords"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, det["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
    print(det["label"])

print(f"Device: {device.upper()} | Entities Found: {zone_count}")

if zone_count > MAX_PLAYROOM_CAPACITY:
    ALERT_TEXT = "OVERCROWDING ALERT"
    cv2.putText(frame, ALERT_TEXT, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # Send alert to google cloud
    # Send_To_Google_Cloud(ALERT_TEXT)

cv2.imwrite("output_detection.jpg", frame)