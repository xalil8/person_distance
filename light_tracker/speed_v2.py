import os
import torch
import cv2
import math
import numpy as np
from ssl import _create_unverified_context
from time import time
from collections import defaultdict, deque
from trackers.multi_tracker_zoo import create_tracker
import telegram


class SpeedTracker:
    def __init__(self, bot_token, chat_id, source_video_path, model_path, polygon_points, speed_limit, tracker):
        _create_default_https_context = _create_unverified_context

        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id

        self.tracker = tracker
        self.temp_time = {}  # Initialize the car_no_change_count dictionary
        self.source_video_path = source_video_path

        self.model_path = model_path

        self.desired_fps = 16
        self.polygon_points = np.array(polygon_points[0])
        self.polygon_points2 = np.array(polygon_points[1])

        self.polygon_points3 = np.array(polygon_points[2])
        self.polygon_points4 = np.array(polygon_points[3])

        self.speed_limit = speed_limit
        self.car_flags = {}
        self.speeds = {}
        self.ss = {}

        self.video_cap = cv2.VideoCapture(self.source_video_path)
        print("CODE HAS STARTED")
        # CAR DETECTION MODEL
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path, force_reload=False, device=device)
        self.names = self.model.names
        self.model.conf = 0.7

        # TRACKER
        self.tracker_name = tracker
        self.tracker_list = create_tracker(
            f'{self.tracker_name}', f"trackers/{self.tracker_name}/configs/{self.tracker_name}.yaml",
            "weights/osnet_x0_25_msmt17.pt", device=torch.device(device), half=False)

        # Initialize the car dictionary
        self.car_dict = defaultdict(deque)

    def reconnect_video(self, video_cap):
        video_cap.release()
        video_cap = cv2.VideoCapture(self.source_video_path)
        return video_cap
    
    def clear_data(self):
        # Clear car related dictionaries and lists
        self.car_flags = {}
        self.speeds = {}
        self.ss = {}
        self.car_dict.clear()


    def process(self):
        count = 0
        prev_time = time()
        car_counter = 0
        while self.video_cap.isOpened():

            ret, frame = self.video_cap.read()
            if not ret:
                raise Exception("Error reading frame")
            count += 1
            if count % 2 != 0:
                continue

            if car_counter > 80:
                self.clear_data()
                car_counter = 0 

            curr_time = time()
            elapsed_time = curr_time - prev_time # elapsed_time is time that frame takes 
            prev_time = curr_time


            fps = 1.0 / elapsed_time
            scaled_fps = (self.desired_fps / fps)

            cv2.putText(frame, f"FPS: {int(fps)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 229, 204), 3)

            results = self.model(frame)
            det = results.xyxy[0]
            
            if det is not None and len(det):
                outputs = self.tracker_list.update(det.cpu(), frame)

                for j, (output) in enumerate(outputs):
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2, c, id = int(x1), int(y1), int(x2), int(y2), int(cls), int(id)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    entering_area = cv2.pointPolygonTest(np.int32([self.polygon_points]), ((center_x, center_y)),False)
                    leaving_area = cv2.pointPolygonTest(np.int32([self.polygon_points2]), ((center_x, center_y)),False)

                    entering_area2 = cv2.pointPolygonTest(np.int32([self.polygon_points4]), ((center_x, center_y)),False)
                    leaving_area2 = cv2.pointPolygonTest(np.int32([self.polygon_points3]), ((center_x, center_y)),False)



                    if entering_area >= 0:  # Car inside or on the "in" line
                        if id not in self.car_dict:  # New car in the area
                            car_counter += 1
                            self.car_flags[id] = "in"  # Set the flag to "init" for the new car
                            self.car_dict[id] = time()
                    elif leaving_area >= 0:  # Car on or outside the "out" line
                        if self.car_flags.get(id) == "in":
                            self.car_flags[id] = "out"  # Set the flag to "out" for the car after leaving the area
                            total_time_spent = time() - self.car_dict.get(id) 
                            self.speeds[id] = (8 / total_time_spent) * 5 * scaled_fps  # Store the speed in the speeds dictionary

                    if leaving_area2 >= 0:  # Car inside or on the "in" line
                        car_counter += 1
                        if id not in self.car_dict:  # New car in the area
                            car_counter += 1
                            self.car_flags[id] = "in2"  # Set the flag to "init" for the new car
                            self.car_dict[id] = time()

                    elif entering_area2 >= 0:  # Car on or outside the "out" line
                        if self.car_flags.get(id) == "in2":
                            self.car_flags[id] = "out2"  # Set the flag to "out" for the car after leaving the area
                            total_time_spent = time() - self.car_dict.get(id) 
                            self.speeds[id] = (8 / total_time_spent) * 5 * scaled_fps # Store the speed in the speeds dictionary           

                    if self.car_flags.get(id) == "out" or self.car_flags.get(id) == "out2":
                        cv2.putText(frame, f"car{id} speed {round(self.speeds[id], 1)} km/h", (x1 + 10, y2 + 20),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                        if self.speeds[id] > self.speed_limit:
                            if self.ss.get(id) == "sended":
                                continue

                            ##################################### 
                            cv2.rectangle(frame, (x1, y1), (x2, y2),  (255, 0, 255), 2)
                            cv2.putText(frame, f"{str(int(conf*100))}  car{id}", (x1+10 , y1-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.circle(frame, (center_x, center_y), radius=3, color=(50, 255, 50), thickness=-1)
                            ##################################### 
                            cv2.polylines(frame, np.int32([self.polygon_points]), True, (55, 155, 255), 3)
                            cv2.polylines(frame, np.int32([self.polygon_points2]), True, (55, 155, 255), 3)
                            cv2.polylines(frame, np.int32([self.polygon_points3]), True, (55, 155, 255), 3)
                            cv2.polylines(frame, np.int32([self.polygon_points4]), True, (55, 155, 255), 3)

                            self.ss[id] = "sended"
                            cv2.imwrite("speed.jpg", frame)
                            self.bot.send_photo(chat_id=self.chat_id, photo=open("speed.jpg", 'rb'),caption="Hızlı Araç Geçişi")


            if cv2.waitKey(16) == ord('q'):
                break

        # Release resources
        self.video_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = "bytetrack"
    speed_limit = 30

    bot_token = "6386993261:AAHViMHwILkOh3zdSG1u6J4nIsrcmT8DJ_Y"
    chat_id = "-1001517789528"
    #bot_token = "6329653365:AAHdjlQmxokp_3kRJxHG74XmCmN4UCRul9w"
    #chat_id = "-1001787996915"
    #source_video_path = "data2.mp4"
    source_video_path="rtsp://admin:esb12345@15.25.110.132:554/cam/realmonitor?channel=1@subtype=0"
    model_path = "speed_v1.pt"

    left_in = [[1136, 152], [725, 152], [735, 134], [1134, 129]]
    left_out = [[1163, 274], [600, 282], [590, 306], [1167, 299]]
    right_in = [[1269, 268], [1275, 296], [1727, 293], [1714, 268]]
    right_out = [[1222, 150], [1526, 150], [1513, 131], [1218, 131]]
    polygon_points = [left_in,left_out,right_in,right_out]

    tracker = SpeedTracker(bot_token, chat_id, source_video_path, model_path, polygon_points, speed_limit, tracker)
    tracker.process()
