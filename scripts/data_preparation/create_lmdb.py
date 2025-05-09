# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
# 
# This script creates LMDB files for the DAVIDE dataset (Train split only).
# This file are used during training for the sake of faster data loading.
#
# The following data types are created:
# - train_gt.lmdb - Ground truth
# - train_blurred.lmdb - Blurred
# - train_depth.lmdb - Depth
# - train_conf_depth.lmdb - Confidence map
# - train_mono_depth_sharp.lmdb - Mono-depth sharp
# - train_mono_depth_blur.lmdb - Mono-depth blur
#
# The script receives the following arguments:
# - dataset_path: Path to the dataset
# - n_thread: Number of threads to use for multiprocessing
#
# Example:
#   python scripts/data_preparation/create_lmdb.py --dataset_path /path/to/dataset --n_thread 4
#
# ------------------------------------------------------------------------
import argparse
import os

from basicsr.utils.misc import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_img_data


def create_lmdb_for_divide(dataset_path, n_thread):
    """
    Create LMDB files for the DAVIDE dataset (Train split only).

    Args:
        dataset_path (str): Path to the dataset.
        n_thread (int): Number of threads to use for multiprocessing.

    This function creates LMDB files for various data types in the DAVIDE dataset, including GT, blur, depth, 
    confidence map, and mono-depth (sharp and blur).
    """
    # train GT
    input_path = os.path.join(dataset_path, 'train', 'gt')
    lmdb_path = os.path.join(dataset_path, 'train_gt.lmdb') 
    img_path_list, keys = prepare_keys_divide(input_path)
    make_lmdb_from_img_data(input_path, lmdb_path, img_path_list, keys, multiprocessing_read=True, n_thread=n_thread)

    # train blur
    input_path = os.path.join(dataset_path, 'train', 'blur')
    lmdb_path = os.path.join(dataset_path, 'train_blurred.lmdb')
    img_path_list, keys = prepare_keys_divide(input_path)
    make_lmdb_from_img_data(input_path, lmdb_path, img_path_list, keys, multiprocessing_read=True, n_thread=n_thread)

    # train depth
    input_path = os.path.join(dataset_path, 'train', 'depth')
    lmdb_path = os.path.join(dataset_path, 'train_depth.lmdb')
    img_path_list, keys = prepare_keys_divide(input_path)
    make_lmdb_from_img_data(input_path, lmdb_path, img_path_list, keys, multiprocessing_read=True, n_thread=n_thread, compress_level=0, input_type='depth')

    # train confidence map
    input_path = os.path.join(dataset_path, 'train', 'conf-depth')
    lmdb_path = os.path.join(dataset_path, 'train_conf_depth.lmdb')
    img_path_list, keys = prepare_keys_divide(input_path)
    make_lmdb_from_img_data(input_path, lmdb_path, img_path_list, keys, multiprocessing_read=True, n_thread=n_thread, compress_level=0, input_type='conf')

    # train mono-depth (sharp)
    input_path = os.path.join(dataset_path, 'train', 'mono-depth_sharp')
    lmdb_path = os.path.join(dataset_path, 'train_mono_depth_sharp.lmdb')
    img_path_list, keys = prepare_keys_divide(input_path)
    make_lmdb_from_img_data(input_path, lmdb_path, img_path_list, keys, multiprocessing_read=True, n_thread=n_thread)

    # train mono-depth (sharp)
    input_path = os.path.join(dataset_path, 'train', 'mono-depth_blur')
    lmdb_path = os.path.join(dataset_path, 'train_mono_depth_blur.lmdb')
    img_path_list, keys = prepare_keys_divide(input_path)
    make_lmdb_from_img_data(input_path, lmdb_path, img_path_list, keys, multiprocessing_read=True, n_thread=n_thread)


def prepare_keys_divide(folder_path):
    """Prepare image path list and keys for the dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help=("Path to DAVIDE dataset"))
    parser.add_argument('--n_thread', type=int, help=("Number of threads to use") )
    args = parser.parse_args()

    create_lmdb_for_divide(args.dataset_path, args.n_thread)
