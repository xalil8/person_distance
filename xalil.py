import torch
import cv2
import math
import numpy as np
from ssl import _create_unverified_context
import time
from collections import defaultdict, deque
from trackers.multi_tracker_zoo import create_tracker

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "custom", path="v6.pt", device="mps")
model.conf = 0.5
class_names = model.names
print(class_names)
model.classes = [0]

tracker_list = create_tracker(f'ocsort', f"trackers/ocsort/configs/ocsort.yaml","weights/osnet_x0_25_msmt17.pt", device=torch.device("mps"), half=False)

# Open the video capture
source_video_path = "distance.mp4"
video_cap = cv2.VideoCapture(source_video_path)

# Define the distance threshold for considering persons as close
distance_threshold = 500  # in pixels

# Define time intervals for checking proximity
check_interval = 15 * 60  # 15 minutes in seconds

# Create a dictionary to store person pairs and their proximity durations
person_pairs = defaultdict(list)

# Define a function to check and process person pairs
def process_person_pairs():
    current_time = time.time()
    pairs_to_remove = []  # Create a list to store pairs for removal
    
    for pair, time_list in person_pairs.items():
        total_time = sum(time_list)
        if total_time >= 5 * 60:  # 5 minutes
            id1, id2 = pair.split('-')
            print(f"Persons {id1} and {id2} spent {total_time:.2f} seconds together.")
            pairs_to_remove.append(pair)  # Add pair to removal list
            
    # Remove pairs outside the loop
    for pair in pairs_to_remove:
        del person_pairs[pair]
fps = 20

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

while video_cap.isOpened():
    last_time = time.time()

    ret, frame = video_cap.read()
    if not ret:
        break

    tracking_list = []

    results = model(frame)
    det = results.xyxy[0]

    pairs= []
    if det is not None and len(det):
        outputs = tracker_list.update(det.cpu(), frame)
        for j, (output) in enumerate(outputs):
            bbox = output[0:4]
            id = output[4]
            cls = output[5]
            conf = output[6]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2, c, id = int(x1), int(y1), int(x2), int(y2), int(cls), int(id)
            center_x, center_y = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
            pairs.append[id,(center_x,center_y)]
 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2),  (255, 0, 255), 2)
            cv2.putText(frame, f"{str(int(conf*100))}  person{id}", (x1+10 , y1-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    print(pairs) 
        # Update person_pairs dictionary based on tracked persons' proximity
        # person_ids = [int(output[4]) for output in outputs]
        # for i, id1 in enumerate(person_ids):
        #     for id2 in person_ids[i+1:]:
        #         pair_key = f"{id1}-{id2}"z
        #         # Extract bounding box information from the output for id1 and id2
        #         box1_centers = int((output[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)

        #         bbox_id1 = outputs[i][:4]
        #         bbox_id2 = outputs[i+1][:4]
                
        #         distance = np.linalg.norm(np.array(bbox_id1)[:2] - np.array(bbox_id2)[:2])

        #         if distance <= distance_threshold:
        #             person_pairs[pair_key].append(1 / fps)  # Add frame duration to pair
        #             # Draw a line between the persons
        #             cv2.line(frame, (int(bbox_id1[0]), int(bbox_id1[1])), (int(bbox_id2[0]), int(bbox_id2[1])), (0, 255, 0), 6)
        #             # Write distance on the line
        #             cv2.putText(frame, f"{distance:.2f}", (int((bbox_id1[0] + bbox_id2[0]) / 2), int((bbox_id1[1] + bbox_id2[1]) / 2)),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6)
        #         else:
        #             cv2.line(frame, (int(bbox_id1[0]), int(bbox_id1[1])), (int(bbox_id2[0]), int(bbox_id2[1])), (0, 0, 255), 6)
        #             # Write distance on the line
        #             cv2.putText(frame, f"{distance:.2f}", (int((bbox_id1[0] + bbox_id2[0]) / 2), int((bbox_id1[1] + bbox_id2[1]) / 2)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)




    # Process person pairs every check_interval seconds
    # if int(time.time() - last_time) % check_interval == 0:
    #     process_person_pairs()

    # # Display total spent time between person pairs on top right of the screen
    # total_time_text = "Total Time: "
    # for pair, time_list in person_pairs.items():
    #     total_time = sum(time_list)
    #     total_time_text += f"{pair}: {total_time:.2f}s||||||  "
    # cv2.putText(frame, total_time_text, (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 5)

    # fps = int(1 / (time.time() - last_time))
    # cv2.putText(frame, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209), 4)
    # frame = cv2.resize(frame, (800, 520))

    cv2.imshow("Frame", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
