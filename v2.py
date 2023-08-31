import torch
import cv2
import time
import math
from light_tracker.tracker import Tracker
# Import the Tracker class

    # ... (same Tracker code as you provided)

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="v5.pt", device="cuda:0")
names = model.names
print(names)
model.conf = 0.4
model.classes = [0]

# Open the video file
video_path = "distance.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Create a VideoWriter to save the processed video
output_path = "outy.mp4"
#out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
print("code has started")

# Create an instance of the Tracker class
tracker = Tracker()

while True:
    last_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))
    
    # Detect objects using YOLOv5
    results = model(frame)
    det = results.xyxy[0]
    
    objects_rect = []
    if det is not None and len(det):
        for j, output in enumerate(det):
            bbox = output[0:4]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            objects_rect.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # Update the tracker and get tracked objects
    tracked_objects = tracker.update(objects_rect)
    
    for tracked_obj in tracked_objects:
        x, y, w, h, obj_id = tracked_obj
        cv2.putText(frame, f"ID: {obj_id}", (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)



    fps = int(1 / (time.time() - last_time))
    cv2.putText(frame, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209), 4)
    frame = cv2.resize(frame, (720, 540))

    cv2.imshow("Video Processing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
