# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------


import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
from os import path as osp

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info

__all__ = ['create_dataset', 'create_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'basicsr.data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
            **kwargs: Other keyword arguments for dataset class.
    
    Returns:
        torch.utils.data.Dataset: Dataset.
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    # log dataset info
    logger = get_root_logger()
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
        'is created.')
    
    return dataset


def create_dataloader(dataset,
                      dataset_opt,
                      num_gpu=1,
                      dist=False,
                      sampler=None,
                      seed=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train', 'val', or 'test'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
            pin_memory (bool, optional): If True, the data loader will copy Tensors 
                into CUDA pinned memory before returning them. Default: False.
            concat_frames_in_batch (bool, optional): If True, use the custom collate 
                function `concat_frames_in_batch_collate_fn` from the dataset. Default: False.
            prefetch_mode (str, optional): Prefetch mode. Can be 'cpu' for CPUPrefetcher, 
                'cuda' for CUDAPrefetcher, or None for normal DataLoader. Default: None.
            num_prefetch_queue (int, optional): Number of prefetch queues. Used only if 
                `prefetch_mode` is 'cpu'. Default: 1.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None

    Returns:
        torch.utils.data.DataLoader or PrefetchDataLoader: The created dataloader.
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    # check if dataset obj has collate_fn
    if dataset_opt.get('concat_frames_in_batch', False):
        dataloader_args['collate_fn'] = dataset.concat_frames_in_batch_collate_fn

    # prefetch mode
    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        # return prefetch dataloader
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # return normal dataloader
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
