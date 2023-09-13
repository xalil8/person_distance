import telegram
import torch
import cv2
import math
import numpy as np
import time
from collections import defaultdict, deque
from trackers.multi_tracker_zoo import create_tracker
import itertools

class ObjectTrackingSystem:
    def __init__(self, source_video_path, surface_polygon,bot_token,chat_id,visualize):
        self.visualize = visualize
        self.source_video_path = source_video_path
        self.surface_polygon = surface_polygon
        self.width =  1280
        self.length = 720
        self.last_photo_sent_time = 0

        ############ Telegram parameters ############
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id
        self.bot.send_message(chat_id=self.chat_id,text="Person distance has started")
        ############################################

        ###########main algorithm parameters########
        # checking last window_size of frame, if more than ,
        # disappear ratio of these frame person close each other, will get error 
        self.window_size = 50 #number of frames that checks for algorithm 
        self.disappear_limit = 200
        self.algorithm_ratio = 0.9 # %90 of frames should be pairs together
        # Dictionary to store sliding windows for each person pair
        self.sliding_windows = {}
        self.counter = 0
        self.distance_threshold = 1000  # in pixels
        ##########################################
        
        ###### perspective transform parameters for bir eye view######
        dst_pts = np.array([[0, 0], [self.width, 0], [0, self.length], [self.width, self.length]], dtype=np.float32)
        self.H = cv2.getPerspectiveTransform( self.surface_polygon, dst_pts)
        self.initialize_model()

    def initialize_model(self):
        # Load the detection model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5m", device="mps")
        self.model.conf = 0.5
        self.class_names = self.model.names
        self.model.classes = [0]
        
        self.tracker_list = create_tracker(f'ocsort', f"trackers/ocsort/configs/ocsort.yaml", "weights/osnet_x0_25_msmt17.pt", device=torch.device("mps"), half=False)


    def point_transform(self, x_old, y_old):
        # Create the homogeneous coordinate for the source point
        source_point = np.array([x_old, y_old, 1])
        # Perform the transformation
        transformed_point = np.dot(self.H, source_point)
        # Normalize the transformed point
        x_new = int(transformed_point[0] / transformed_point[2])
        y_new = int(transformed_point[1] / transformed_point[2])
        return int(x_new), int(y_new)

    # check person pairs, if any pairs doesnt show up for a more than disappear limit
    # that pair deleted from memmory 
    def clean_dict(self, main_dict, count):
        keys_to_delete = []  # Create a list to store keys to delete

        for pair_key, pair_data in main_dict.items():
            if (count - main_dict[pair_key]["last_count"] > self.disappear_limit):
                keys_to_delete.append(pair_key)
                print(f"{pair_key} CLEANED")
        # Delete items outside the loop
        for key in keys_to_delete:
            del main_dict[key]
        return main_dict


    def run(self):
        video_cap = cv2.VideoCapture(self.source_video_path)
        counter = 0

        try :
            while video_cap.isOpened():
                counter += 1 
                last_time = time.time()
                ret, frame = video_cap.read()
                if not ret:
                    break
                if counter % 2 != 0:
                    continue
                tracking_list = []
                frame = cv2.resize(frame, (self.width, self.length))

                results = self.model(frame)
                det = results.xyxy[0]

                bird_eye_view2  = np.zeros((self.length, self.width, 3), dtype=np.uint8)
                
                if det is not None and len(det):
                    outputs = self.tracker_list.update(det.cpu(), frame)
                    ids = outputs[:, 4]

                    unique_ids = np.unique(ids)
                    for id1, id2 in itertools.combinations(unique_ids, 2):
                        id1, id2 = int(id1), int(id2)  # Convert IDs to integers
                        pair_key = f"{id1}-{id2}"

                        bbox1 = outputs[ids == id1][0, :4]
                        bbox2 = outputs[ids == id2][0, :4]

                        center1 = ((bbox1[0] + bbox1[2]) / 2, bbox1[3])
                        center2 = ((bbox2[0] + bbox2[2]) / 2, bbox2[3])

                        transformed_center1 = self.point_transform(center1[0], center1[1])
                        transformed_center2 = self.point_transform(center2[0], center2[1])
                        transformed_distance = int(np.linalg.norm(np.array(transformed_center1) - np.array(transformed_center2)))

                        if pair_key not in self.sliding_windows:
                            self.sliding_windows[pair_key] = {"elements": deque(maxlen=self.window_size), "last_count": None}

                        self.sliding_windows[pair_key]["last_count"] = counter

                        if transformed_distance < self.distance_threshold:
                            color = (0, 0, 255)
                            self.sliding_windows[pair_key]["elements"].append(True)
                        else:
                            color = (0, 255, 255)
                            self.sliding_windows[pair_key]["elements"].append(False)
                    

                        if len(self.sliding_windows[pair_key]["elements"]) == self.window_size and \
                                self.sliding_windows[pair_key]["elements"].count(True) / self.window_size >= self.algorithm_ratio:
                                    
                            current_time = time.time()
                            time_since_last_photo = current_time - self.last_photo_sent_time
                            if time_since_last_photo >= 30:  # 60 seconds
                                print("so close")
                                resized = cv2.resize(frame,(1280,720))
                                cv2.imwrite("distance.jpg",resized)
                                self.bot.send_photo(chat_id=self.chat_id, photo=open("distance.jpg", 'rb'), caption="CALISANLAR BERABER VAKIT GECIRIYOR")
                                self.last_photo_sent_time = current_time  # Update the last photo sent time

                        #visualization on 
###########################################################################################

                        if self.visualize:
                            for j, (x1, y1, x2, y2, id, cls, conf) in enumerate(outputs[:, :7]):
                                x1, y1, x2, y2, id = map(int, (x1, y1, x2, y2, id))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                cv2.putText(frame, f"{str(int(conf * 100))}  person{id}", (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8, (0, 0, 255), 2)
                            distance = int(np.linalg.norm(np.array(center1) - np.array(center2)))

                            cv2.circle(bird_eye_view2, (int(transformed_center1[0]), int(transformed_center1[1])), 5, (0, 0, 255), 3)
                            cv2.circle(bird_eye_view2, (int(transformed_center2[0]), int(transformed_center2[1])), 5, (0, 0, 255), 3)

                            cv2.putText(frame, f"{distance}", (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            cv2.line(frame, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), color, 6)
                            #### transformed
                            cv2.putText(bird_eye_view2, f"{transformed_distance}", (int((transformed_center1[0] + transformed_center2[0]) / 2), int((transformed_center1[1] + transformed_center2[1]) / 2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            cv2.line(bird_eye_view2, (int(transformed_center1[0]), int(transformed_center1[1])), 
                                    (int(transformed_center2[0]), int(transformed_center2[1])), (200, 222, 20), 6)
###########################################################################################

                fps = int(1 / (time.time() - last_time))
                print(self.sliding_windows)
                
                # every 200 frame check person pairs to clean memory
                if (counter % 200) == 0:
                    print("CLEANING CHECKED ")
                    self.sliding_windows = self.clean_dict(self.sliding_windows, counter)


                if self.visualize:
                    cv2.putText(bird_eye_view2, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209), 4)
                    bird_eye_view2 = cv2.resize(bird_eye_view2, (self.width, self.length))
                    stacked = np.hstack((frame, bird_eye_view2))
                    cv2.imshow("Frame1", stacked)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            video_cap.release()
            cv2.destroyAllWindows()
            
    
    
if __name__ == "__main__":
    bot_token = "6329653365:AAHdjlQmxokp_3kRJxHG74XmCmN4UCRul9w"
    chat_id = "-1001787996915"

    source_video_path = "sari_yelek_v8.mp4"
    visuliaze = False
    surface_polygon = np.array([[483, 170], [759, 162], [30, 606], [1273, 608]], dtype=np.float32)
    tracker_system = ObjectTrackingSystem(source_video_path, surface_polygon,bot_token,chat_id,visuliaze)
    tracker_system.run()
