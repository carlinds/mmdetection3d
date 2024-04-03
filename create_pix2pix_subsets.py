import os
import random
import shutil

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

DATASET_FRACTION = 0.75

# Set the source and destination directories
source_dir = '/proj/agp/renders/real2sim/nusc_pix2pixhd/samples'
destination_dir = f'/proj/agp/renders/real2sim/nusc_pix2pixhd_{DATASET_FRACTION}/samples'

# Define the folders
folders = [
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT',
    'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
]

with open('random_indices.txt', 'r') as f:
    random_indices = [int(line.strip()) for line in f.readlines()]

# Select a fraction of the images
num_images = 40157
random_indices = random_indices[:int(DATASET_FRACTION * num_images)]

# Iterate through each folder
for folder in folders:
    # Get the list of files in the folder
    files = os.listdir(os.path.join(source_dir, folder))
    files = sorted(files)
    files = [files[i] for i in random_indices]

    # Create the destination folder if it doesn't exist
    destination_folder = os.path.join(destination_dir, folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    def _copy_file(file):
        source_file = os.path.join(source_dir, folder, file)
        destination_file = os.path.join(destination_folder, file)

        # Skip if the file already exists
        if os.path.exists(destination_file):
            return

        # If you want to copy files, use shutil.copy:
        shutil.copy(source_file, destination_file)
        # If you want to symlink files, use os.symlink:
        #os.symlink(source_file, destination_file)

    # Copy or symlink selected files to the destination folder
    process_map(_copy_file, files, max_workers=8)
