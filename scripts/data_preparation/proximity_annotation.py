# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
#
# This script generates proximity labels for the DAVIDE dataset (test split). The proximity labels
# are based on the depth images. The depth images are read from the dataset and the proximity
# labels are calculated based on the number of pixels in the close, mid, and far ranges.
#
# The script generates a CSV file with the following columns:
# - clip: Name of the clip
# - frame: Name of the frame
# - close: Proximity label for close range
# - mid: Proximity label for mid range
# - far: Proximity label for far range
#
# The script receives the following arguments:
# - root: Path to the depth folder
# - csv: Path to the CSV file to save the proximity values
# - n_threads: Number of threads to use
#
# Example:
#   python scripts/data_preparation/proximity_annotation.py --root /path/to/depth --csv /path/to/proximity.csv --n_threads 10
#
# ------------------------------------------------------------------------
import pandas as pd
import os
import numpy as np
from basicsr.utils import FileClient, imfrombytes
from multiprocessing import Pool
import argparse
from tqdm import tqdm


# Function to get pairs of subdirectories and files
def get_subdirs_files(root_dir):
    subdirs = []
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        for f in sorted(filenames):
            subdirs.append(os.path.basename(root))
            filename_without_ext = os.path.splitext(os.path.basename(f))[0]
            files.append(filename_without_ext)
    # Sort both lists based on the file names
    subdirs, files = zip(*sorted(zip(subdirs, files)))
    return list(subdirs), list(files)


# This class is used to read the depth images from the dataset
class DepthReader:
    def __init__(self, root):
        self.root = root
        self.file_client = FileClient(backend='disk')

    def __call__(self, clip, frame):
        path = os.path.join(self.root, clip, frame + '.png')
        depth_bytes = self.file_client.get(path)
        depth = imfrombytes(depth_bytes, datatype='depth', float32=True)
        return depth
    

# Function to get the proximity value based on the depth image
def get_proximity(depth):
    # Count the number of pixels in the close, mid, and far ranges
    close_pixels = np.sum(depth < 1.5)
    mid_pixels = np.sum((depth >= 1.5) & (depth < 4.5))
    far_pixels = np.sum(depth >= 4.5)
    # get label based on the number of pixels
    proximity = {
        'close': 0,
        'mid': 0,
        'far': 0
    }
    if close_pixels > mid_pixels and close_pixels > far_pixels:
        proximity['close'] = 1
    elif mid_pixels > close_pixels and mid_pixels > far_pixels:
        proximity['mid'] = 1
    else:
        proximity['far'] = 1
    return proximity


# Function to process a file
def process_file(subdir_file_root,):
    subdir, file, root = subdir_file_root
    # Create a DepthReader object
    depth_reader = DepthReader(root)
    # Read the depth image
    depth = depth_reader(subdir, file)
    # Calculate the proximity value
    proximity = get_proximity(depth)
    # Combine dictionaries
    info = {**{'clip': subdir, 'frame': file}, **proximity}
    return info
    

def main(root, n_threads, csv_file):
    
    # Get the subdirectories and files
    subdirs, files = get_subdirs_files(root)
    roots = [root] * len(subdirs)

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'clip': subdirs,
        'frame': files
    })

    df['close'] = np.nan
    df['mid'] = np.nan
    df['far'] = np.nan
    
    # Create a list to store the results
    results = []
    # Create a multiprocessing Pool
    with Pool(processes=n_threads) as pool:
        # Use pool.map to apply process_file to each file in parallel
        for info in tqdm(pool.imap_unordered(process_file, zip(subdirs, files, roots)), total=len(subdirs), desc="Processing files"):
            results.append(info)

    # Append the results to the dataframe
    for info in results:
        df.loc[(df['clip'] == info['clip']) & (df['frame'] == info['frame']), 'close'] = info['close']
        df.loc[(df['clip'] == info['clip']) & (df['frame'] == info['frame']), 'mid'] = info['mid']
        df.loc[(df['clip'] == info['clip']) & (df['frame'] == info['frame']), 'far'] = info['far']

    # Save the dataframe to a CSV file
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help=("Path to the depth folder"))
    parser.add_argument('--csv', type=str, help=("Path to the CSV file to save the proximity values to"))
    parser.add_argument('--n_threads', type=int, help=("Number of threads to use"))
    args = parser.parse_args()
    main(args.root, args.n_threads, args.csv)