import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import subprocess
import threading
import time
import queue

# =========================
# Configuration
# =========================

RTMP_URL = os.environ.get("RTMP_URL")

STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_FPS = 15
FRAME_DELAY = 1.0 / STREAM_FPS  

# --- Zone Split (left = daycare, right = restricted) ---
# The screen is split exactly down the middle on the X axis.
DAYCARE_ZONE_RATIO    = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
RESTRICTED_ZONE_RATIO = np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]])
 
# Alert thresholds
DAYCARE_CROWD_THRESHOLD = 3       # "Crowded" when pet count EXCEEDS this value
ALERT_COOLDOWN_SEC      = 3.0     # Seconds between repeated on-screen alerts

# ZONE_RATIOS = np.array([
#    [0.0, 0.0],
#    [1.0, 0.0],
#    [1.0, 1.0],
#    [0.0, 1.0]
# ])

frame_queue = queue.Queue(maxsize=30)
classifier_queue = queue.Queue(maxsize=2) 

running = True

# Thread-safe storage mapped directly to persistent Tracking IDs
latest_breeds = {}
breed_lock = threading.Lock()

# Alert state (guarded by alert_lock)
alert_lock            = threading.Lock()
alert_restricted_ts   = 0.0   # last time a "restricted zone" alert was shown
alert_crowded_ts      = 0.0   # last time a "crowded" alert was shown

def start_ffmpeg_stream():
   if RTMP_URL is None:
       print("ERROR: RTMP_URL environment variable is not set.")
       exit(1)

   ffmpeg_cmd = [
       "ffmpeg",
       "-y",
       "-f", "rawvideo",
       "-pix_fmt", "yuv420p",
       "-s", f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
       "-r", str(STREAM_FPS),
       "-i", "-",
       "-f", "lavfi",
       "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
       "-c:v", "libx264", 
       "-preset", "ultrafast",         
       "-tune", "zerolatency",          
       "-b:v", "1000k",                 
       "-g", str(STREAM_FPS * 2),       
       "-c:a", "aac",
       "-b:a", "128k",
       "-ar", "48000",
       "-map", "0:v:0",
       "-map", "1:a:0",
       "-f", "flv",
       "-flvflags", "no_duration_filesize",
       RTMP_URL
   ]

   print("[FFmpeg] Initializing streamlined frame pipeline...")
   return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=1024*1024)


# =========================
# Background Streaming Thread
# =========================
def stream_worker():
   global running
   ffmpeg_process = start_ffmpeg_stream()
   print("[Stream Worker] Streaming thread active.")
   
   while running:
       start_time = time.time()
       try:
           frame = frame_queue.get(timeout=0.1)
       except queue.Empty:
           continue

       try:
           ffmpeg_process.stdin.write(frame.tobytes())
           ffmpeg_process.stdin.flush()
       except BrokenPipeError:
           print("\n[Stream Worker] Pipeline disconnected by cloud server endpoint.")
           running = False
           break

       elapsed = time.time() - start_time
       sleep_time = FRAME_DELAY - elapsed
       if sleep_time > 0:
           time.sleep(sleep_time)

   print("[Stream Worker] Cleaning up streaming pipes...")
   if ffmpeg_process.stdin:
       ffmpeg_process.stdin.close()
   ffmpeg_process.wait()


# =========================
# Sequential GPU Initialization (THE FIX)
# =========================
print("Initializing TensorRT Engines on GPU...")

# Load both models sequentially on the main thread to prevent context deadlocks
detector_model = YOLO("yolov8n.engine", task="detect")
classifier_model = YOLO("yolov8n-cls.engine", task="classify")

print("Warming up CUDA Core Graphs...")
dummy_blank_frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)

with torch.no_grad():
    _ = detector_model(dummy_blank_frame, verbose=False)
    _ = classifier_model(dummy_crop, imgsz=224, verbose=False)
    
print("GPU Engines ready and execution contexts mapped.")

# =========================
# Asynchronous GPU Breed Classifier Thread
# =========================
def breed_classifier_worker():
    global running
    print("[Breed Worker] Initializing TensorRT Classifier Engine...")
    classifier_model = YOLO("yolov8n-cls.engine", task="classify")
    
    dummy_blank_frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
    with torch.no_grad():
        _ = classifier_model(dummy_blank_frame, imgsz=224, verbose=False)
    print("[Breed Worker] TensorRT Classifier running completely isolated on GPU.")

    # Hardcoded ImageNet Dog Breeds (IDs 151 to 268)
    IMAGENET_DOG_CLASSES = [
        "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", 
        "Papillon", "Toy terrier", "Rhodesian ridgeback", "Afghan hound", "Basset", "Beagle", 
        "Bloodhound", "Bluetick", "Black-and-tan coonhound", "Walker hound", "English foxhound", 
        "Redbone", "Borzoi", "Irish wolfhound", "Italian greyhound", "Whippet", "Ibizan hound", 
        "Norwegian elkhound", "Otterhound", "Saluki", "Scottish deerhound", "Weimaraner", 
        "Staffordshire bullterrier", "American Staffordshire terrier", "Bedlington terrier", 
        "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier", 
        "Yorkshire terrier", "Wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier", 
        "Airedale", "Cairn", "Australian terrier", "Dandie Dinmont", "Boston bull", 
        "Miniature schnauzer", "Giant schnauzer", "Standard schnauzer", "Scotch terrier", 
        "Tibetan terrier", "Silky terrier", "Soft-coated wheaten terrier", 
        "West Highland white terrier", "Lhasa", "Flat-coated retriever", "Curly-coated retriever", 
        "Golden retriever", "Labrador retriever", "Chesapeake Bay retriever", 
        "German short-haired pointer", "Vizsla", "English setter", "Irish setter", "Gordon setter", 
        "Brittany spaniel", "Clumber", "English springer", "Welsh springer spaniel", "Cocker spaniel", 
        "Sussex spaniel", "Irish water spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", 
        "Briard", "Kelpie", "Komondor", "Old English sheepdog", "Shetland sheepdog", "Collie", 
        "Border collie", "Bouvier des Flandres", "Rottweiler", "German Shepherd", "Doberman", 
        "Miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller", 
        "EntleBucher", "Boxer", "Bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane", 
        "Saint Bernard", "Eskimo dog", "Malamute", "Siberian husky", "Dalmatian", "Affenpinscher", 
        "Basenji", "Pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", 
        "Chow", "Keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "Toy poodle", 
        "Miniature poodle", "Standard poodle", "Mexican hairless"
    ]

    while running:
        try:
            crop_item = classifier_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        crop = crop_item["crop"]
        track_id = crop_item["id"]

        try:
            with torch.no_grad():
                cls_results = classifier_model(crop, imgsz=224, verbose=False)

            if cls_results and cls_results[0].probs is not None:
                res = cls_results[0]
                prob = res.probs.top1conf.item()
                class_id = res.probs.top1
                
                if prob > 0.4:
                    # Check if the predicted ID falls within the Dog section of ImageNet
                    if 151 <= class_id <= 268:
                        # Map the integer back to the string
                        breed_name = IMAGENET_DOG_CLASSES[class_id - 151]
                    else:
                        # Fallback just in case it classifies the dog as a generic mammal/wolf
                        breed_name = "Unknown Dog Breed"
                        
                    label = f"{breed_name} ({prob:.2f})"
                else:
                    label = "Dog: Breed Unclear"
            else:
                label = "Dog: Identifying..."

            with breed_lock:
                latest_breeds[track_id] = label

        except Exception as e:
            print(f"\n[Breed Worker ERROR] Classification failed: {e}")

        finally:
            classifier_queue.task_done()

# =========================
# Alert Overlay Helper
# =========================
 
def draw_alert_banner(frame, message, color, y_offset=0):
    overlay = frame.copy()
    banner_h = 44
    y_start  = y_offset
    y_end    = y_offset + banner_h
    cv2.rectangle(overlay, (0, y_start), (frame.shape[1], y_end), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame, message,
        (10, y_start + 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )

# =========================
# Start Camera & Workers
# =========================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, STREAM_FPS)

if not cap.isOpened():
   print("ERROR: Could not open webcam.")
   exit(1)

packer_thread = threading.Thread(target=stream_worker, daemon=True)
packer_thread.start()

classifier_thread = threading.Thread(target=breed_classifier_worker, daemon=True)
classifier_thread.start()


# =========================
# Main AI Processing Loop
# =========================

print("Processing frames cleanly. Stream running smoothly on GPU. Press 'q' to quit.")

try:
    while cap.isOpened() and running:
        success, frame = cap.read()
        if not success:
            break
 
        frame   = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
        frame_h, frame_w = frame.shape[:2]
        now     = time.time()
 
        # ── Build zone polygon coordinates ──────────────────────────────────
        daycare_zone    = (DAYCARE_ZONE_RATIO    * [frame_w, frame_h]).astype(np.int32)
        restricted_zone = (RESTRICTED_ZONE_RATIO * [frame_w, frame_h]).astype(np.int32)
 
        # ── Draw zone backgrounds (semi-transparent tint) ───────────────────
        overlay = frame.copy()
        cv2.fillPoly(overlay, [daycare_zone],    (0, 180, 0))    # green tint  – daycare
        cv2.fillPoly(overlay, [restricted_zone], (0, 0, 200))    # red tint    – restricted
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
 
        # ── Draw zone borders & labels ───────────────────────────────────────
        cv2.polylines(frame, [daycare_zone],    True, (0, 220, 0),  2)
        cv2.polylines(frame, [restricted_zone], True, (0, 0, 255),  2)
 
        mid_x = frame_w // 2
        cv2.putText(frame, "DAYCARE ZONE",    (10,       frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0),  2, cv2.LINE_AA)
        cv2.putText(frame, "RESTRICTED ZONE", (mid_x + 6, frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2, cv2.LINE_AA)
 
        # ── Run YOLO tracker ─────────────────────────────────────────────────
        results = detector_model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )
 
        daycare_pet_count    = 0
        restricted_pet_count = 0
        detection_list       = []
 
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
 
            for box, track_id_tensor in zip(r.boxes, r.boxes.id):
                class_id = int(box.cls[0])
                track_id = int(track_id_tensor.item())
 
                # class 0 = person, class 16 = dog  (COCO)
                if class_id not in [0, 16]:
                    continue
 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bottom_center   = (int((x1 + x2) / 2), y2)
 
                in_daycare    = cv2.pointPolygonTest(daycare_zone,    bottom_center, False) >= 0
                in_restricted = cv2.pointPolygonTest(restricted_zone, bottom_center, False) >= 0
 
                if not (in_daycare or in_restricted):
                    continue   # outside both zones
 
                # ── Zone accounting (pets only, not people) ──────────────────
                is_pet = (class_id == 16)
                if is_pet:
                    if in_daycare:
                        daycare_pet_count += 1
                    elif in_restricted:
                        restricted_pet_count += 1
 
                # ── Build display label ──────────────────────────────────────
                if class_id == 0:
                    display_label = f"Person [ID:{track_id}]"
                else:
                    with breed_lock:
                        breed_status = latest_breeds.get(track_id)
 
                    if breed_status is not None:
                        display_label = f"ID {track_id}: {breed_status}"
                    else:
                        display_label = f"ID {track_id}: Analyzing..."
                        if not classifier_queue.full():
                            pad  = 10
                            crop = frame[
                                max(0, y1 - pad):min(frame_h, y2 + pad),
                                max(0, x1 - pad):min(frame_w, x2 + pad),
                            ]
                            if crop.size > 0:
                                classifier_queue.put_nowait({"crop": crop, "id": track_id})
 
                # ── Dot & zone indicator ─────────────────────────────────────
                dot_color = (0, 0, 255) if in_restricted else (0, 200, 0)
                cv2.circle(frame, bottom_center, 6, dot_color, -1)
 
                detection_list.append({
                    "coords":       (x1, y1, x2, y2),
                    "label":        display_label,
                    "in_restricted": in_restricted,
                })
 
        # ── Draw bounding boxes ───────────────────────────────────────────────
        for det in detection_list:
            x1, y1, x2, y2 = det["coords"]
            box_color  = (0, 0, 255) if det["in_restricted"] else (255, 120, 0)
            text_color = (0, 40, 255) if det["in_restricted"] else (0, 150, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, det["label"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 2, cv2.LINE_AA)
 
        # ── HUD counters ─────────────────────────────────────────────────────
        cv2.putText(frame, f"Daycare pets: {daycare_pet_count}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Restricted pets: {restricted_pet_count}",
                    (mid_x + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2, cv2.LINE_AA)
 
        # ── Alert banners ─────────────────────────────────────────────────────
 
        if restricted_pet_count > 0:
            with alert_lock:
                show = (now - alert_restricted_ts) >= ALERT_COOLDOWN_SEC or alert_restricted_ts == 0
                if show:
                    alert_restricted_ts = now
            msg = (f"ALERT: {restricted_pet_count} PET(S) IN RESTRICTED ZONE!"
                   if restricted_pet_count > 1
                   else "ALERT: PET DETECTED IN RESTRICTED ZONE!")
            draw_alert_banner(frame, msg, (0, 0, 200), y_offset=banner_y)
            banner_y += 46
 
        if daycare_pet_count > DAYCARE_CROWD_THRESHOLD:
            with alert_lock:
                show = (now - alert_crowded_ts) >= ALERT_COOLDOWN_SEC or alert_crowded_ts == 0
                if show:
                    alert_crowded_ts = now
            msg = f"CROWDED: {daycare_pet_count} PETS IN DAYCARE ZONE (limit {DAYCARE_CROWD_THRESHOLD})"
            draw_alert_banner(frame, msg, (0, 130, 200), y_offset=banner_y)
 
        cv2.imshow("Jetson Pet Detection", frame)
 
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(yuv_frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
   print("Shutting down cleanly...")
   running = False
   cap.release()
   cv2.destroyAllWindows()
   packer_thread.join(timeout=3.0)
   print("Done.")
