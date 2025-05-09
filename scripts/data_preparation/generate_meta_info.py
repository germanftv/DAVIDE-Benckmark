# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
#
# This script generates meta_info files for the DAVIDE dataset.
# The meta_info files are used during training and testing to load the data.
#
# The script receives the following arguments:
# - dataset_path: Path to the DAVIDE dataset
#
# The script generates two meta_info files:
# - meta_info_DAVIDE_train.txt  # Train split
# - meta_info_DAVIDE_test.txt   # Test split
#
# Example:
#   python scripts/data_preparation/generate_meta_info.py --dataset_path /path/to/dataset
#
# ------------------------------------------------------------------------
import os
import glob
import argparse
from tqdm import tqdm

INTERVAL = 4            # Index interval between adjacent frames in DAVIDE dataset

def generate_meta_info_txt(data_path, split, meta_info_path):
    """Generate meta_info txt file for the DAVIDE dataset.

    Args:
        data_path (str): Path to the DAVIDE dataset.
        split (str): 'train' or 'test'.
        meta_info_path (str): Path to save the meta_info txt file.

    The format is:
        "video_name num_frames (height,width,channel) start_frame interval"
    """
    f= open(meta_info_path, "w+")
    file_list = sorted(glob.glob(os.path.join(data_path, split, 'gt/*')))
    total_frames = 0
    for path in tqdm(file_list):
        name = os.path.basename(path)
        frames = sorted(glob.glob(os.path.join(path, '*')))
        start_frame = os.path.basename(frames[0]).split('.')[0]
        total_frames += len(frames)

        f.write(f"{name} {len(frames)} (1440,1920,3) {start_frame} {INTERVAL}\r\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate meta_info files for DAVIDE dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help= 'Path to DAVIDE dataset')
    args = parser.parse_args()

    # Generate meta_info txt for train and test splits
    generate_meta_info_txt(args.dataset_path, 'train', 'dataset/meta_info/meta_info_DAVIDE_train.txt')
    generate_meta_info_txt(args.dataset_path, 'test', 'dataset/meta_info/meta_info_DAVIDE_test.txt')


