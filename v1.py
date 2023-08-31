import torch
import cv2
import time 

model = torch.hub.load("ultralytics/yolov5", "custom", path="sasa_yelek.pt", device="cuda:0")
names = model.names
model.conf = 0.4
model.classes = [1]
# Open the video file
video_path = "distance.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)


# Create a VideoWriter to save the processed video
output_path = "outy.mp4"
#out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
print("code has started")
while True:
    last_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1920, 1080))
    results = model(frame)
    det = results.xyxy[0]
    
    if det is not None and len(det):
        for j, output in enumerate(det):
            bbox = output[0:4]
            conf = output[4]
            id = int(output[5])
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{names[id]} {conf:.2f}", (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
    #out.write(frame)
    fps = int(1 / (time.time() - last_time))
    
    cv2.putText(frame, f"{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 20, 209),4)
    frame = cv2.resize(frame, (720, 540))
    
    cv2.imshow("Video Processing", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
