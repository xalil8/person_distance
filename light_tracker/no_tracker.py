import cv2
import numpy as np
import torch
import time

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "custom", path="v4.pt", force_reload=False, device="mps")
model.conf = 0.8
class_names = model.names
#model.classes = [2]

# Open the video capture
source_video_path = "v1.mp4"
video_cap = cv2.VideoCapture(source_video_path)

prev_time = time.time()
frame_count = 0

while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame)
    det = results.xyxy[0]

    if det is not None and len(det):
        for j, (output) in enumerate(det):
            bbox = output[0:4]
            conf = round(float(output[4]), 2)
            class_id = int(output[5])
            class_name = class_names[class_id]

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = f"{class_name} - {conf}"
            cv2.putText(frame, text, (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Calculate and display FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = frame_count / elapsed_time
    frame_count = 0  # Reset frame count
    prev_time = current_time  # Reset previous time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
