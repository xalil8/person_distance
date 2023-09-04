import torch
import cv2
import numpy as np
import time
import itertools

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "yolov5m", device="mps")
model.conf = 0.5
model.classes = [0]

# Open the video capture
source_video_path = "distance.mp4"
video_cap = cv2.VideoCapture(source_video_path)

# Define the distance threshold for considering persons as close
distance_threshold = 400  # in pixels

# Sliding window parameters
window_size = 100  # Number of frames in the sliding window
threshold_range = 60  # Threshold for % of frames spent together

# Create a dictionary to store person pairs and their frame counts
pair_frames = {}

while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break

    results = model(frame)
    det = results.xyxy[0]

    if det is not None and len(det):
        outputs = det.cpu().numpy()

        # Extract the IDs from the outputs
        ids = outputs[:, 5].astype(int)

        # Loop through all unique pairs of IDs
        unique_ids = np.unique(ids)
        for id1, id2 in itertools.combinations(unique_ids, 2):
            id1, id2 = int(id1), int(id2)  # Convert IDs to integers
            pair_key = f"{id1}-{id2}"

            # Find the corresponding bounding boxes for id1 and id2
            bbox1 = outputs[ids == id1][0, :4]
            bbox2 = outputs[ids == id2][0, :4]

            center
