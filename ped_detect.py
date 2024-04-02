import os, sys
import argparse
from ultralytics import YOLO
import torch
# import math
import time, datetime
import cv2

torch.cuda.set_device(0) # Set to your desired GPU number
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Loading YOLO Model")

model = YOLO("yolov8n.pt")
if torch.cuda.is_available(): model.to(device=device)

zfill_amount = 10 # this is used to zero-pad the frame integer once converted to string (e.g. 0000000001, 0000000002)

def resize(image, scale_factor = 50):
    width = int(image.shape[1] * scale_factor / 100)
    height = int(image.shape[0] * scale_factor / 100)
    dimension = (width, height)
    # resize image
    resized_image = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
    return resized_image

if __name__ == "__main__":
    program_start = time.time()
    parser = argparse.ArgumentParser(description='Detect people in a video')
    parser.add_argument('-v', '--video_file', type=str, help='The path to the video file')

    # sys.argv.extend(["-v", "convertedExterndisk0_Ch2_20240305170000_20240305180000Converted.mp4"])
    # sys.argv.extend(["-v", "vid.webm"])

    args = parser.parse_args()
  
    filename = args.video_file
    cap = cv2.VideoCapture(filename)
    print(f"Loaded video {filename}")

    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_counter = 0 # to count how many times the model inferenced (to be used for file saving)
    frame_counter = 0 # to count for progress

    save_path = f"results/{filename}/"
    if not os.path.exists(save_path): os.makedirs(save_path) #creates a dedicated folder for the video file
    log_path = f"logs/{filename}"
    if not os.path.exists(log_path): os.makedirs(log_path)

    save_path_pattern = os.path.join(save_path, f"{filename}_%{str(zfill_amount)}d") #not used for now
    print(save_path_pattern)

    print("Running model inference")

    inference_start = time.time()

    while cap.isOpened():
        # Reading the video stream
        ret, image = cap.read()
        if ret:
            results = model(image, verbose=False)
            frame_counter += 1
            if frame_counter % 500 == 0: print(f"{frame_counter} out of {frame_length} frames at {round((frame_counter / frame_length) * 100, 2)}%", end="\r")

            for r in results:
                boxes = r.boxes
                if 0 in boxes.cls:  # If there is a person/s (index 0) in the frame
                    inference_counter += 1
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0: #Only get person boxes

                            # bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                            # put box in cam
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                            # confidence
                            # confidence = math.ceil((box.conf[0]*100))/100

                            # object details
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2

                            cv2.putText(image, "person", org, font, fontScale, color, thickness)
                image = resize(image)
                cv2.imwrite(os.path.join(save_path, f"{filename}_{str(inference_counter).zfill(zfill_amount)}.jpg"), image)  #zfill will zero-pad integer with {zfill_amount} digits (e.g. 001, 002)
        else:
            print(f"Video Stream is done at {inference_counter} inferred frames out of {frame_counter} actual frames")
            break
    end = time.time()

    log = [
        f"Frames Inferred: {inference_counter}",
        f"Actual Frame Count: {frame_counter}",
        f"Inference Percentage: {round((inference_counter / frame_counter) * 100, 2)}%",
        f"Total Program Time: {end-program_start}",
        f"Inference Only Time: {end-inference_start}"
    ]

    print(log, sep="\n")

    ts = time.time()
    sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H_%M_%S')

    #Write results to log folder
    write_path = os.path.join(log_path, f"{filename}_{sttime}_log.txt")
    results_log = open(write_path, "w", encoding="utf-8")
    results_log.write(str(log))
    results_log.close()
    cap.release()