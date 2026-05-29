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

ZONE_RATIOS = np.array([
   [0.0, 0.0],
   [1.0, 0.0],
   [1.0, 1.0],
   [0.0, 1.0]
])

frame_queue = queue.Queue(maxsize=30)
classifier_queue = queue.Queue(maxsize=2) 

running = True

# Thread-safe storage mapped directly to persistent Tracking IDs
latest_breeds = {}
breed_lock = threading.Lock()

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

       frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
       frame_h, frame_w = frame.shape[:2]
       focus_zone_coords = (ZONE_RATIOS * [frame_w, frame_h]).astype(np.int32)
       ZONES = {"focus_zone": focus_zone_coords}

       results = detector_model.track(
           frame,
           persist=True, 
           tracker="bytetrack.yaml", 
           verbose=False
       )

       zone_count = 0
       detection_list = []
       cv2.polylines(frame, [ZONES["focus_zone"]], True, (0, 255, 255), 2)

       for r in results:
           if r.boxes is not None and r.boxes.id is not None:
               for box, track_id_tensor in zip(r.boxes, r.boxes.id):
                   class_id = int(box.cls[0])
                   track_id = int(track_id_tensor.item())

                   if class_id in [0,16]:
                       x1, y1, x2, y2 = map(int, box.xyxy[0])
                       bottom_center = (int((x1 + x2) / 2), y2)

                       if cv2.pointPolygonTest(ZONES["focus_zone"], bottom_center, False) >= 0:
                           zone_count += 1
                           cv2.circle(frame, bottom_center, 6, (0, 0, 255), -1)

                           if class_id == 0:
                               display_label = f"Person [ID: {track_id}]"
                           else:
                               with breed_lock:
                                   breed_status = latest_breeds.get(track_id)

                               if breed_status is not None:
                                   display_label = f"ID {track_id}: {breed_status}"
                               else:
                                   display_label = f"ID {track_id}: Analyzing Breed..."
                                   
                                   if not classifier_queue.full():
                                       pad = 10
                                       crop = frame[
                                           max(0, y1 - pad):min(frame_h, y2 + pad),
                                           max(0, x1 - pad):min(frame_w, x2 + pad)
                                       ]
                                       if crop.size > 0:
                                           classifier_queue.put_nowait({"crop": crop, "id": track_id})

                           detection_list.append({
                               "coords": (x1, y1, x2, y2),
                               "label": display_label
                           })

       for det in detection_list:
           x1, y1, x2, y2 = det["coords"]
           cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
           cv2.putText(frame, det["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

       cv2.putText(frame, f"Entities: {zone_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
