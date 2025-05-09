# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import functools
import os
import socket
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(launcher, backend='nccl', **kwargs):
    """
    Initialize the distributed computing environment.

    Args:
        launcher (str): Launcher type. Options are "pytorch" and "slurm".
        backend (str, optional): Backend of torch.distributed. Defaults to 'nccl'.
        **kwargs: Additional arguments for torch.distributed.init_process_group.
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
        hostname = socket.gethostname()
        rank = dist.get_rank() + 1
        gpu = f"[{hostname}-{rank}]"
        world_size = dist.get_world_size()
        print(f"{gpu} is OK (global rank: {rank}/{world_size})")
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
        hostname = socket.gethostname()
        local_rank = int(os.environ["LOCAL_RANK"])
        gpu = f"[{hostname}-{local_rank}]"
        rank = dist.get_rank() + 1
        world_size = dist.get_world_size()
        print(f"{gpu} is OK (global rank: {rank}/{world_size})")
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    """
    Initialize the PyTorch distributed environment.

    Args:
        backend (str): Backend of torch.distributed.
        **kwargs: Additional keyword arguments for dist.init_process_group.
    """
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    """
    Get the rank and world size for the current distributed process.

    Returns:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_rank():
    """
    Get the rank of the current process.

    Returns:
        int: Rank of the current process. Returns 0 if torch.distributed is not available or not initialized.
    """
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def get_world_size():
    """
    Get the total number of processes in the current distributed environment.

    Returns:
        int: Number of processes. Returns 1 if torch.distributed is not available or not initialized.
    """
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def get_local_rank():
    """
    Get the rank of the current process within its node group.

    Returns:
        int: Rank of the current process within its node group. Returns 0 if torch.distributed is not available or not initialized. 
             If the 'LOCAL_RANK' environment variable is set, it returns its value. Otherwise, it returns the global rank.
    """
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    if os.environ.get('LOCAL_RANK') is not None:
        return int(os.environ['LOCAL_RANK'])
    else:
        return dist.get_rank()      # When using single node, local rank is the same as global rank. 


def master_only(func):
    """
    Decorator to make a function only run on the master process.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function that only runs on the master process.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
