import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import json
import datetime
import tempfile

from google.cloud import storage, pubsub_v1
from twilio.rest import Client as TwilioClient

# Configuration
MAX_PLAYROOM_CAPACITY = 3
IMAGE_PATH = "pet_daycare3.jpg"
ALERT_TEXT = ""

DEVICE_ID = os.environ.get("DEVICE_ID", "jetson-nano-01")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "cs131detections")
PUBSUB_TOPIC = os.environ.get("PUBSUB_TOPIC", "projects/cs131-final-project-497022/topics/detections")

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")
ALERT_TO = os.environ.get("ALERT_TO")

# Define dynamic zone ratios to cover the ENTIRE image
ZONE_RATIOS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])


def upload_snapshot_to_gcs(frame: np.ndarray, alert_label: str) -> str:
    try:
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"alerts/{DEVICE_ID}/snapshot_{alert_label.replace(' ', '_')}_{timestamp}.jpg"

        # Write frame to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, frame)

        # Upload to GCS
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        blob.upload_from_filename(tmp_path, content_type="image/jpeg")
        os.remove(tmp_path)

        gcs_uri = f"gs://{GCS_BUCKET}/{filename}"
        print(f"[GCS] Snapshot uploaded: {gcs_uri}")
        return gcs_uri

    except Exception as e:
        print(f"[GCS] Upload failed: {e}")
        return "unavailable"


def generate_signed_url(gcs_uri: str, expiration_minutes: int = 60) -> str:
    try:
        # Parse bucket and blob from gs://bucket/path
        path = gcs_uri.replace("gs://", "")
        bucket_name = path.split("/")[0]
        blob_name = "/".join(path.split("/")[1:])

        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_minutes * 60,
            method="GET"
        )
        return signed_url

    except Exception as e:
        print(f"[GCS] Could not generate signed URL: {e}")
        return gcs_uri  # fallback to raw GCS path


def publish_to_pubsub(alert_label: str, zone_count: int, gcs_uri: str, detections: list):
    try:
        publisher = pubsub_v1.PublisherClient()

        payload = {
            "device_id": DEVICE_ID,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "alert": {
                "label": alert_label,
                "zone_count": zone_count,
                "max_capacity": MAX_PLAYROOM_CAPACITY,
                "snapshot_url": gcs_uri,
                "detections": detections  # list of detected labels
            }
        }

        future = publisher.publish(PUBSUB_TOPIC, json.dumps(payload).encode("utf-8"))
        print(f"[PubSub] Alert published. Message ID: {future.result()}")

    except Exception as e:
        print(f"[PubSub] Publish failed: {e}")


def send_twilio_sms(alert_label: str, zone_count: int, signed_url: str, detections: list):
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO]):
        print("[Twilio] Missing env vars — skipping direct SMS.")
        return

    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)

        detection_summary = "\n".join(f"  • {d}" for d in detections) or "  • None"

        body = (
            f"{alert_label}\n"
            f"──────────────────\n"
            f"Device   : {DEVICE_ID}\n"
            f"Count    : {zone_count}/{MAX_PLAYROOM_CAPACITY} max\n"
            f"Time     : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"Detected :\n{detection_summary}\n"
            f"Snapshot : {signed_url}"
        )

        message = client.messages.create(body=body, from_=TWILIO_FROM, to=ALERT_TO)
        print(f"[Twilio] SMS sent! SID: {message.sid}")

    except Exception as e:
        print(f"[Twilio] SMS failed: {e}")


def send_alert(frame: np.ndarray, alert_label: str, zone_count: int, detections: list):
    print(f"\n[ALERT] {alert_label} — triggering alert pipeline...")

    gcs_uri = upload_snapshot_to_gcs(frame, alert_label)
    signed_url = generate_signed_url(gcs_uri)

    publish_to_pubsub(alert_label, zone_count, gcs_uri, detections)
    send_twilio_sms(alert_label, zone_count, signed_url, detections)

    print(f"[ALERT] Pipeline complete.\n")


print("Loading Optimized Deep Learning Models...")
#device = "mps" if torch.backends.mps.is_available() else 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
detector_model = YOLO("yolov8n.pt")
classifier_model = YOLO("yolov8m-cls.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Check your webcam connection")
        break
    frame_h, frame_w = frame.shape[:2]
    focus_zone_coords = (ZONE_RATIOS * [frame_w, frame_h]).astype(np.int32)
    ZONES = {"focus_zone": focus_zone_coords}

    # Run detector with half-precision (if it detects a CUDA compatible GPU)
    results = detector_model(frame, stream=True, half=(device == 'cuda'))
    dog_count = 0

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

                    cv2.circle(frame, bottom_center, 6, (0, 0, 255), -1)
                    detection_data = {"coords": (x1, y1, x2, y2), "class": class_name, "label": class_name.title()}

                    if class_name == "dog":
                        dog_count += 1
                        pad = 10
                        crop = frame[max(0, y1 - pad):min(frame_h, y2 + pad), max(0, x1 - pad):min(frame_w, x2 + pad)]
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

    print(f"Device: {device.upper()} | Number of Dogs: {dog_count}")

    if dog_count > MAX_PLAYROOM_CAPACITY:
        ALERT_TEXT = "OVERCROWDING ALERT"
        cv2.putText(frame, ALERT_TEXT, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        detected_labels = [det["label"] for det in detection_list]

        # Send alert to google cloud
        # Send_To_Google_Cloud(ALERT_TEXT)
        send_alert(
            frame=frame,
            alert_label=ALERT_TEXT,
            zone_count=dog_count,
            detections=detected_labels
        )
    cv2.imshow("Jetson Pet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()