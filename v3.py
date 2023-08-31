import os
import torch
import cv2
import math
import numpy as np
from ssl import _create_unverified_context
import time 
from collections import defaultdict, deque
from trackers.multi_tracker_zoo import create_tracker

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "custom", path="v5.pt", device="cuda:0")
model.conf = 0.5
class_names = model.names
model.classes = [0]

#tracker_list = create_tracker(f'bytetrack', f"trackers/bytetrack/configs/bytetrack.yaml","weights/osnet_x0_25_msmt17.pt", device=torch.device("cuda:0"), half=False)

tracker_list = create_tracker(f'ocsort', f"trackers/ocsort/configs/ocsort.yaml","weights/osnet_x0_25_msmt17.pt", device=torch.device("cuda:0"), half=False)

# Open the video capture
source_video_path = "distance.mp4"
video_cap = cv2.VideoCapture(source_video_path)


while video_cap.isOpened():
    last_time = time.time()

    ret, frame = video_cap.read()
    if not ret:
        break
    
    tracking_list = []

    results = model(frame)
    det = results.xyxy[0]
    
    if det is not None and len(det):
        outputs = tracker_list.update(det.cpu(), frame)

        for j, (output) in enumerate(outputs):
            bbox = output[0:4]
            id = output[4]
            cls = output[5]
            conf = output[6]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2, c, id = int(x1), int(y1), int(x2), int(y2), int(cls), int(id)

            cv2.rectangle(frame, (x1, y1), (x2, y2),  (255, 0, 255), 2)
            cv2.putText(frame, f"{str(int(conf*100))}  person{id}", (x1+10 , y1-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # text = f"{class_name} - Confidence: {conf}"
            # cv2.putText(frame, text, (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    print((time.time() - last_time))
    fps = int(1 / (time.time() - last_time))
    cv2.putText(frame, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209), 4)
    frame = cv2.resize(frame, (1080, 720))
    
    cv2.imshow("Frame", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()


