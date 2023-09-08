import torch
import cv2
import math
import numpy as np
import time
from collections import defaultdict, deque
from trackers.multi_tracker_zoo import create_tracker
import itertools

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "yolov5m", device="cuda:0")
model.conf = 0.5
class_names = model.names
model.classes = [0]
tracker_list = create_tracker(f'ocsort', f"trackers/ocsort/configs/ocsort.yaml", "weights/osnet_x0_25_msmt17.pt", device=torch.device("cuda:0"), half=False)


def clean_dict(main_dict,count):
    
    keys_to_delete = []  # Create a list to store keys to delete

    for pair_key, pair_data in main_dict.items():
        if (count - sliding_windows[pair_key]["last_count"] > disappear_limit):
            keys_to_delete.append(pair_key)
            print(f"{pair_key} CLEANED")
    # Delete items outside the loop
    for key in keys_to_delete:
        del main_dict[key]
    return main_dict

# Open the video capture
source_video_path = "sari_yelek_v8.mp4"
video_cap = cv2.VideoCapture(source_video_path)

# Define the distance threshold for considering persons as close
distance_threshold = 400  # in pixels

# Define time intervals for checking proximity
check_interval = 15 * 60  # 15 minutes in seconds

# Define the sliding window size
window_size = 30
disappear_limit = 50
# Dictionary to store sliding windows for each person pair
sliding_windows = {}
counter = 0
while video_cap.isOpened():
    counter += 1 
    last_time = time.time()

    ret, frame = video_cap.read()
    if not ret:
        break
    
    tracking_list = []

    results = model(frame)
    det = results.xyxy[0]

    if det is not None and len(det):
        outputs = tracker_list.update(det.cpu(), frame)

        # Extract the IDs from the outputs
        ids = outputs[:, 4]

        # Loop through all unique pairs of IDs
        unique_ids = np.unique(ids)
        for id1, id2 in itertools.combinations(unique_ids, 2):
            id1, id2 = int(id1), int(id2)  # Convert IDs to integers
            pair_key = f"{id1}-{id2}"

            # Find the corresponding bounding boxes for id1 and id2
            bbox1 = outputs[ids == id1][0, :4]
            bbox2 = outputs[ids == id2][0, :4]

            center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

            # Calculate the Euclidean distance between the centers
            distance = int(np.linalg.norm(np.array(center1) - np.array(center2)))

            if pair_key not in sliding_windows:
                sliding_windows[pair_key] = {"elements":deque(maxlen=window_size),"last_count":None}
                #sliding_windows[pair_key] = deque(maxlen=window_size)

            sliding_windows[pair_key]["last_count"] = counter
            # Label the frame based on distance
            if distance < distance_threshold:
                color = (0, 0, 255)
                sliding_windows[pair_key]["elements"].append(True)
            else:
                color = (0, 255, 255)
                sliding_windows[pair_key]["elements"].append(False)

            # Check if %80 of the 100 elements in the sliding window are True
            if len(sliding_windows[pair_key]) == window_size and sliding_windows[pair_key].count(True) / window_size >= 0.9:
                # Add your desired logic here
                pass
            
            cv2.putText(frame, f"{distance}", (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.line(frame, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), color, 6)

        for j, (x1, y1, x2, y2, id, cls, conf) in enumerate(outputs[:, :7]):
            x1, y1, x2, y2, id = map(int, (x1, y1, x2, y2, id))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{str(int(conf * 100))}  person{id}", (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        #print(sliding_windows["1-3"]["elements"])
    fps = int(1 / (time.time() - last_time))
    cv2.putText(frame, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209), 4)
    frame = cv2.resize(frame, (800, 520))



    if (counter %100) == 0:
        print("CLEANING CHECKED ")
        sliding_windows = clean_dict(sliding_windows,counter)
            
    for pair_key, pair_data in sliding_windows.items():
        print(counter,sliding_windows[pair_key]["last_count"])
    print("//////////////////////////////////////////////////")
    
    
    cv2.imshow("Frame", frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()

