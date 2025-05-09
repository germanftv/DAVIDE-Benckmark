# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import numpy as np
import os
import random
import time
import torch
from os import path as osp
from contextlib import contextmanager

from basicsr.utils.dist_util import master_only


class ProgressBar:
    """
    Progress bar class for tracking the progress of a task.

    Attributes:
        pbar (tqdm.tqdm): TQDM progress bar instance.
    """
    @master_only
    def __init__(self, total, dataset_name) -> None:
        """
        Initialize the ProgressBar.

        Args:
            total (int): Total number of iterations or tasks.
            dataset_name (str): Name of the dataset being processed.
        """
        from tqdm import tqdm

        self.pbar = tqdm(total=total)
        self.pbar.set_description(f'Test on {dataset_name}')

    @master_only
    def update(self):
        """
        Update the progress bar by one step.
        """
        self.pbar.update(1)
    
    @master_only
    def close(self):
        """
        Close the progress bar.
        """
        self.pbar.close()


@contextmanager
def nullcontext(enter_result=None):
    """Context manager that does nothing.

    This context manager is useful as a stand-in for code that optionally uses a context manager. It yields the 
    provided `enter_result` and does not perform any additional operations.

    Args:
        enter_result (Any): The value to be yielded by the context manager. Default is None.

    Yields:
        Any: The `enter_result` value.
    """
    yield enter_result


def set_random_seed(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to set for the random number generators.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch (both CPU and CUDA) to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    """
    Get the current time as a formatted string.

    Returns:
        str: The current time formatted as 'YYYYMMDD_HHMMSS'.
    """
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """
    Create a directory.
    
    The function checks if the specified path exists. If it does, it renames the existing directory by appending a timestamp to its name and then creates a new directory with the original name.

    Args:
        path (str): The path of the directory to create.

    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def tensor_min(x:torch.Tensor, dims:tuple, keepdim=True):
    """
    Return the minimum value of a tensor along the specified dimensions.

    Args:
        x (torch.Tensor): The input tensor.
        dims (tuple): A tuple of dimensions along which to compute the minimum.
        keepdim (bool, optional): Whether to retain the reduced dimensions in the output tensor. Default is True.

    Returns:
        torch.Tensor: The tensor containing the minimum values along the specified dimensions.
    """
    if len(dims) > 1 and type(dims) == tuple:
        return tensor_min(tensor_min(x, dims[-1:], keepdim=keepdim), dims[0:-1], keepdim=keepdim)
    else:
        return torch.min(x, dims[0], keepdim=keepdim).values


def tensor_max(x:torch.Tensor, dims:tuple, keepdim=True):
    """
    Return the maximum value of a tensor along the specified dimensions.

    Args:
        x (torch.Tensor): The input tensor.
        dims (tuple): A tuple of dimensions along which to compute the maximum.
        keepdim (bool, optional): Whether to retain the reduced dimensions in the output tensor. Default is True.

    Returns:
        torch.Tensor: The tensor containing the maximum values along the specified dimensions.
    """
    if len(dims) > 1 and type(dims) == tuple:
        return tensor_max(tensor_max(x, dims[-1:], keepdim=keepdim), dims[0:-1], keepdim=keepdim)
    else:
        return torch.max(x, dims[0], keepdim=keepdim).values


@master_only
def make_exp_dirs(opt):
    """
    Create directories for experiments.

    This function creates necessary directories for training or testing experiments. 
    If in training mode, it creates an experiment root directory and renames any existing directory with the same name. 
    For testing mode, it creates a results root directory. Additionally, it creates other specified directories, 
    excluding those related to strict loading, pretraining networks, and resuming experiments.

    Args:
        opt (dict): Options dictionary containing paths and configuration for the experiment.

    """
    path_opt = opt['path'].copy()
    if opt['is_train']:
        experiments_root = path_opt.pop('experiments_root')
        experiment_root = osp.join(experiments_root, opt['name'])
        mkdir_and_rename(experiment_root)
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network'
                                           not in key) and ('resume'
                                                            not in key):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def scandir_SIDD(dir_path, keywords=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        keywords (str | tuple(str), optional): File keywords that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (keywords is not None) and not isinstance(keywords, (str, tuple)):
        raise TypeError('"keywords" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, keywords, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if keywords is None:
                    yield return_path
                elif return_path.find(keywords) > 0:
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, keywords=keywords, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, keywords=keywords, recursive=recursive)


def check_resume(opt, resume_iter):
    """
    Check and update resume states and pretrain_network paths.

    This function checks if the resume state is specified in the options.
    If so, it identifies all network keys and checks if any pretrain paths are specified, 
    issuing a warning if they are. It then sets the pretrained model paths for each network, 
    unless the network is listed in the ignore_resume_networks option.
    
    Args:
        opt (dict): Options dictionary containing configuration for the experiment.
        resume_iter (int): The iteration number from which to resume.

    """
    from basicsr.utils.logger import get_root_logger

    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (
                    basename not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = osp.join(
                    opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")


def sizeof_fmt(size, suffix='B'):
    """
    Convert a file size to a human-readable format.
    
    The function iteratively divides the size by 1024 and appends the appropriate unit from bytes (B) to yottabytes (YB) until the size is less than 1024.

    Args:
        size (int): File size in bytes.
        suffix (str): Suffix to append to the size units. Default is 'B'.

    Returns:
        str: Formatted file size as a string with appropriate units (e.g., '10.0 KB', '1.5 MB').
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'
