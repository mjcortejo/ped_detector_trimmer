import os, sys
import glob
import argparse
from ultralytics import YOLO
import torch
import math
import time, datetime
import cv2

import multiprocessing
from tqdm import tqdm

torch.cuda.set_device(0) # Set to your desired GPU number
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Define a function that will be executed in parallel
def process_data(filepath, results_path):
    cap = cv2.VideoCapture(filepath)
    filename = os.path.basename(filepath)

    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_counter = 0 # to count how many times the model inferenced (to be used for file saving)
    frame_counter = 0 # to count for progress

    # save_path = f"results/{filename}/"
    save_path = os.path.join(results_path, filename)
    if not os.path.exists(save_path): os.makedirs(save_path) #creates a dedicated folder for the video file
    log_path = f"logs/"
    if not os.path.exists(log_path): os.makedirs(log_path)

    save_path_pattern = os.path.join(save_path, f"{filename}_%{str(zfill_amount)}d") #not used for now

    inference_start = time.time()

    while cap.isOpened():
        # Reading the video stream
        ret, image = cap.read()
        if ret:
            results = model(image, verbose=False)
            frame_counter += 1
            # if frame_counter % 500 == 0: print(f"{frame_counter} out of {frame_length} frames at {round((frame_counter / frame_length) * 100, 2)}%", end="\r")

            for r in results:
                boxes = r.boxes
                detected = False
                if 0 in boxes.cls:  # If there is a person/s (index 0) in the frame
                    inference_counter += 1
                    detected = True
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
                if detected:
                    # image = resize(image)
                    cv2.imwrite(os.path.join(results_path, f"{filename}_{str(inference_counter).zfill(zfill_amount)}.jpg"), image)  #zfill will zero-pad integer with {zfill_amount} digits (e.g. 001, 002)
                    # save results to the 
        else:
            # print(f"Video {filename} is done at {inference_counter} inferred frames out of {frame_counter} actual frames")
            break
    end = time.time()

    log = [
        f"Frames Inferred: {inference_counter}",
        f"Actual Frame Count: {frame_counter}",
        f"Inference Percentage: {round((inference_counter / frame_counter) * 100, 2)}%",
        # f"Total Program Time: {end-program_start}",
        f"Inference Only Time: {end-inference_start}"
    ]


    ts = time.time()
    sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H_%M_%S')

    #Write results to log folder
    write_path = os.path.join(log_path, f"{filename}_{sttime}_log.txt")
    results_log = open(write_path, "w", encoding="utf-8")
    results_log.write(str(log))
    results_log.close()
    cap.release()


if __name__ == "__main__":
    # program_start = time.time()
    parser = argparse.ArgumentParser(description='Detect people in a video')
    parser.add_argument('-j', '--job_folder', type=str, help='The path to the videos file', default="jobs/")
    parser.add_argument('-e', '--extension', type=str, help='The extension of the videos', default=".mp4")
    parser.add_argument('-n', '--num_workers', type=str, help='Number of workers to use', default=4)
    parser.add_argument('-s', '--save_folder', type=str, help='The path to save the results', default="results/")
    
    # add argument to enable flag for multiprocessing
    parser.add_argument('-m', '--multi', action='store_true', help='Enable multi-processing', default=True)

    args = parser.parse_args()
  
    job_path = args.job_folder + "*" + args.extension
    print(f"Job folder file path is ",job_path)
    
    save_path = args.save_folder
    print(f"Savings results to {save_path}")
    
    video_files = glob.glob(job_path)
    print(f"Found {len(video_files)} files")

    multi_job = args.multi

    if multi_job:
        num_workers = int(args.num_workers)
        print(f"Multi-processing enabled. Will use {num_workers} workers for multiprocessing")

        # Create a multiprocessing pool
        model.share_memory()
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(num_workers)

        # Use tqdm to track the progress of multiprocessing
        results = list(tqdm(pool.imap(process_data, video_files, save_path), total=len(video_files)))

        # Close the pool of processes
        pool.close()
        pool.join()

        # Print the results
        print(f"Done processing {len(results)} files")
    else:
        for filepath in video_files:
            process_data(filepath, save_path)

    

