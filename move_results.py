import os
import shutil
from tqdm import tqdm

origin_path = "D:\\Documents\\Projects\\ped_detector\\results\\"
destination_path = "G:\\My Drive\\Ped Detector Images\\results\\"

finished_folders = os.listdir(origin_path)

print(finished_folders)

for folder in tqdm(finished_folders, "Moving Results folder"):
    full_origin_path = os.path.join(origin_path, folder)
    full_destination_path = os.path.join(destination_path, folder)
    if not os.path.exists(full_destination_path):
        new_path = shutil.copytree(full_origin_path, full_destination_path)

# jobs_path = "D:\\Documents\\Projects\\ped_detector\\jobs"
# jobs_folder = os.listdir(jobs_path)

# for folder in tqdm(finished_folders, "Deleting Jobs folder"):
#     full_jobs_path = os.path.join(jobs_path, folder)
#     if os.path.exists(full_jobs_path):
#         os.remove(full_jobs_path)