import torch
import cv2
import math
import numpy as np
import time
from collections import defaultdict, deque
from trackers.multi_tracker_zoo import create_tracker
import itertools

# Load the detection model
model = torch.hub.load("ultralytics/yolov5", "yolov5m", device="mps")
model.conf = 0.5
class_names = model.names
model.classes = [0]
tracker_list = create_tracker(f'ocsort', f"trackers/ocsort/configs/ocsort.yaml", "weights/osnet_x0_25_msmt17.pt", device=torch.device("mps"), half=False)

#for birdeye view
width,lenght = 1280,720
surface_polygon = np.array([[483, 170], [759, 162], [30, 606], [1273, 608]], dtype=np.float32)
dst_pts = np.array([[0, 0], [width, 0], [0, lenght], [width, lenght]], dtype=np.float32)
H = cv2.getPerspectiveTransform(surface_polygon, dst_pts)


# Open the video capture
source_video_path = "sari_yelek_v8.mp4"
video_cap = cv2.VideoCapture(source_video_path)

# Define the distance threshold for considering persons as close
distance_threshold = 300  # in pixels

# Define time intervals for checking proximity
check_interval = 15 * 60  # 15 minutes in seconds

# Define the sliding window size
window_size = 300
disappear_limit = 100
# Dictionary to store sliding windows for each person pair
sliding_windows = {}
counter = 0

def point_transform(x_old,y_old):
    
    # Create the homogeneous coordinate for the source point
    source_point = np.array([x_old, y_old, 1])
    # Perform the transformation
    transformed_point = np.dot(H, source_point)
    # Normalize the transformed point
    x_new = int(transformed_point[0] / transformed_point[2])
    y_new = int(transformed_point[1] / transformed_point[2])
    return int(x_new),int(y_new) 


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


    

while video_cap.isOpened():
    counter += 1 
    last_time = time.time()
    ret, frame = video_cap.read()
    
    frame = cv2.resize(frame,(width,lenght))
    if not ret:
        break
    
    tracking_list = []

    results = model(frame)
    det = results.xyxy[0]



    # bird_eye_view  = np.zeros((1440, 2560, 3), dtype=np.uint8)
    # cv2.polylines(bird_eye_view, [np.array( [[1046, 424], [649, 1150], [1902, 1115], [1433, 418]], dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    bird_eye_view2  = np.zeros((lenght, width, 3), dtype=np.uint8)
    #cv2.polylines(bird_eye_view2, [np.array([[point_transform(961, 383)], [point_transform(164, 1242)], [point_transform(2493, 1217)], [point_transform(1573, 377)]], dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    

    if det is not None and len(det):
        outputs = tracker_list.update(det.cpu(), frame)
        # Extract the IDs from the outputs
        ids = outputs[:, 4]
        
        for j, (x1, y1, x2, y2, id, cls, conf) in enumerate(outputs[:, :7]):
            x1, y1, x2, y2, id = map(int, (x1, y1, x2, y2, id))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{str(int(conf * 100))}  person{id}", (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # Loop through all unique pairs of IDs
        unique_ids = np.unique(ids)
        for id1, id2 in itertools.combinations(unique_ids, 2):
            id1, id2 = int(id1), int(id2)  # Convert IDs to integers
            pair_key = f"{id1}-{id2}"

            # Find the corresponding bounding boxes for id1 and id2
            bbox1 = outputs[ids == id1][0, :4]
            bbox2 = outputs[ids == id2][0, :4]


            center1 = ((bbox1[0] + bbox1[2]) / 2, bbox1[3])
            center2 = ((bbox2[0] + bbox2[2]) / 2, bbox2[3])
            
            transformed_center1 = point_transform(center1[0],center1[1])
            transformed_center2 = point_transform(center2[0],center2[1])

            transformed_frame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))




            cv2.line(bird_eye_view2, (int(transformed_center1[0]), int(transformed_center1[1])), 
                     (int(transformed_center2[0]), int(transformed_center2[1])), (200,222,20), 6)
            cv2.circle(bird_eye_view2, (int(transformed_center1[0]),int(transformed_center1[1])), 5, (0, 0, 255), 3)
            cv2.circle(bird_eye_view2, (int(transformed_center2[0]),int(transformed_center2[1])), 5, (0, 0, 255), 3)
            
            
            # Calculate the Euclidean distance between the centers
            transformed_distance = int(np.linalg.norm(np.array(transformed_center1) - np.array(transformed_center2)))
            distance = int(np.linalg.norm(np.array(center1) - np.array(center2)))



            if pair_key not in sliding_windows:
                sliding_windows[pair_key] = {"elements":deque(maxlen=window_size),"last_count":None}
                #sliding_windows[pair_key] = deque(maxlen=window_size)

            sliding_windows[pair_key]["last_count"] = counter
            # Label the frame based on distance
            if transformed_distance < distance_threshold:
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
            
            #### transformed
            cv2.putText(transformed_frame, f"{transformed_distance}", (int((transformed_center1[0] + transformed_center2[0]) / 2), int((transformed_center1[1] + transformed_center2[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.line(transformed_frame, (int(transformed_center1[0]), int(transformed_center1[1])), (int(transformed_center2[0]), int(transformed_center2[1])), color, 6)



        #print(sliding_windows["1-3"]["elements"])
    fps = int(1 / (time.time() - last_time))
    cv2.putText(bird_eye_view2, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209), 4)



    if (counter %100) == 0:
        print("CLEANING CHECKED ")
        sliding_windows = clean_dict(sliding_windows,counter)
            
    # for pair_key, pair_data in sliding_windows.items():
    #     print(counter,sliding_windows[pair_key]["last_count"])
    # print("//////////////////////////////////////////////////")
    
    transformed_frame = cv2.resize(transformed_frame, (800, 520))
    frame = cv2.resize(frame, (800, 520))
    
    
    # bird_eye_view = cv2.resize(bird_eye_view, (800, 520))
    # bird_eye_view2 = cv2.resize(bird_eye_view2, (800, 520))
    #stacked = np.hstack((frame,transformed_frame))
    
    
    
    cv2.imshow("Frame1", bird_eye_view2)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()

