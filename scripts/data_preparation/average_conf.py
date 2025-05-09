# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
#
# This script computes the average confidence of the depth images in the DAVIDE dataset (test split).
# The confidence values are read from the dataset and the average confidence is calculated for each
# depth image.
#
# The script generates a CSV file with the following columns:
# - clip: Name of the clip
# - frame: Name of the frame
# - mean_conf: Average confidence value
#
# The script receives the following arguments:
# - root: Path to the conf folder
# - csv: Path to the CSV file to save the mean confidence values
# - n_threads: Number of threads to use
#
# Example:
#   python scripts/data_preparation/average_conf.py --root /path/to/conf --csv /path/to/mean_conf.csv --n_threads 10
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


# This class is used to read the confidence maps from the dataset
class ConfReader:
    def __init__(self, root):
        self.root = root
        self.file_client = FileClient(backend='disk')

    def __call__(self, clip, frame):
        path = os.path.join(self.root, clip, frame + '.png')
        conf_bytes = self.file_client.get(path)
        conf = imfrombytes(conf_bytes, datatype='conf', float32=True)
        return conf
    

# Function to process a file
def process_file(subdir_file_root,):
    subdir, file, root = subdir_file_root
    # Create a ConfReader object
    conf_reader = ConfReader(root)
    # Read the depth image
    conf = conf_reader(subdir, file)
    # Calculate the mean confidence
    mean_conf = conf.mean() * 100
    # Combine dictionaries
    info = {'clip': subdir, 'frame': file, 'mean_conf': mean_conf}
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

    df['mean_conf'] = np.nan
    
    # Create a list to store the results
    results = []
    # Create a multiprocessing Pool
    with Pool(processes=n_threads) as pool:
        # Use pool.map to apply process_file to each file in parallel
        for info in tqdm(pool.imap_unordered(process_file, zip(subdirs, files, roots)), total=len(subdirs), desc="Processing files"):
            results.append(info)

    # Append the results to the dataframe
    for info in results:
        df.loc[(df['clip'] == info['clip']) & (df['frame'] == info['frame']), 'mean_conf'] = info['mean_conf']

    # Save the dataframe to a CSV file
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help=("Path to the conf folder"))
    parser.add_argument('--csv', type=str, help=("Path to the CSV file to save the mean confidence values"))
    parser.add_argument('--n_threads', type=int, help=("Number of threads to use"))
    args = parser.parse_args()
    main(args.root, args.n_threads, args.csv)