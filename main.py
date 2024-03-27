import os
from ultralytics import YOLO
import torch
import math
import cv2

torch.cuda.set_device(0) # Set to your desired GPU number
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Loading YOLO Model")

model = YOLO("yolov8n.pt")
if torch.cuda.is_available(): model.to(device=device)

# detector
filename = "convertedExterndisk0_Ch2_20240305170000_20240305180000Converted.mp4"
cap = cv2.VideoCapture(filename)
print("Loaded video")

frame = 0

save_path = f"results/{filename}/"

if not os.path.exists(save_path): os.makedirs(save_path)
  
while cap.isOpened():
  # Reading the video stream
  ret, image = cap.read()
  results = model(image)
  for r in results:
    boxes = r.boxes

    for box in boxes:
      cls = int(box.cls[0])

      if cls == 0: #If person detected
        # bounding box
        frame += 1 
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        # put box in cam
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # confidence
        confidence = math.ceil((box.conf[0]*100))/100
        # print("Confidence --->",confidence)

        # object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(image, "person", org, font, fontScale, color, thickness)
        cv2.imwrite(os.path.join(save_path, f"{filename}_{frame}.jpg"), image) 
        # Showing the output Image
        # cv2_imshow(image)
        # cv2.imshow("Image", image)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        # else:
        #     break

cap.release()
cv2.destroyAllWindows()