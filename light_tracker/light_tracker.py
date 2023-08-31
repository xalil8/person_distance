import cv2
import numpy as np
import torch
import telegram
from tracker import*
from time import time

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "custom", path="v4.pt", force_reload=False, device="mps")
model.conf = 0.5
class_names = model.names
model.classes = [2]

tracker = Tracker()

# Open the video capture
source_video_path = "v1.mp4"
video_cap = cv2.VideoCapture(source_video_path)

prev_time = time()
frame_count = 0

while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break

    frame_count += 1

    tracking_list = []
    results = model(frame)
    det = results.xyxy[0]
    
    if det is not None and len(det):
        for j, (output) in enumerate(det):
            bbox = output[0:4]
            conf = round(float(output[4]), 2)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            tracking_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(tracking_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 255), 2)  # Draw a purple rectangle around the person
        cv2.putText(frame, f"{id}", (x3 + 10, y3 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Calculate and display FPS
    current_time = time()
    elapsed_time = current_time - prev_time
    fps = frame_count / elapsed_time
    frame_count = 0  # Reset frame count
    prev_time = current_time  # Reset previous time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
