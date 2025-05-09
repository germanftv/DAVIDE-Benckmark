# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------

import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

from basicsr.utils import get_root_logger
from basicsr.data.davide_dataset import DavideTrainDataset


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler.
    Supports enlarging the dataset for iteration-based training, saving
    time when restarting the dataloader after each epoch.

    Attributes:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int): Number of processes participating in the training.
        rank (int): Rank of the current process within num_replicas.
        shuffle (bool): Whether to shuffle the dataset indices.
        epoch (int): Current epoch number.
        num_samples (int): Number of samples to draw for each replica.
        total_size (int): Total size of the dataset after enlarging.

    Methods:
        - __iter__(): Returns an iterator over the indices of the dataset.
        - __len__(): Returns the number of samples.
        - set_epoch(epoch): Sets the epoch number for deterministic shuffling.
    """

    def __init__(self, dataset, num_replicas, rank, shuffle=True, ratio=1):
        """Initializes the sampler.
        
        Args:
            dataset (torch.utils.data.Dataset): Dataset used for sampling.
            num_replicas (int | None): Number of processes participating in
                the training. It is usually the world_size.
            rank (int | None): Rank of the current process within num_replicas.
            shuffle (bool): Whether to shuffle the dataset indices. Default: True.
            ratio (int): Enlarging ratio. Default: 1.
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Sets the current epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.epoch = epoch


class BalancedClipSampler(Sampler):
    """Sampler that takes an equal number of sequences from each clip.

    Attributes:
        dataset (DavideTrainDataset): The dataset being sampled from.
        num_replicas (int): The number of distributed replicas.
        rank (int): The rank of the current process.
        shuffle (bool): Whether to shuffle the samples.
        epoch (int): The current epoch.
        num_seqs_per_video (int): The number of sequences to sample from each clip.
        seed (int): The random seed.
        clip_indices (dict): A dictionary mapping each unique clip to its corresponding indices.
        num_samples (int): The number of samples in the dataset.
        total_size (int): The total size of the dataset.

    Methods:
        - __iter__(): Returns an iterator over the samples.
        - __len__(): Returns the number of samples in the dataset.
        - set_epoch(epoch): Sets the current epoch.

    """
    def __init__(self, dataset:DavideTrainDataset, num_seqs_per_video:int, num_replicas:int=None, rank:int=None, shuffle:bool=True, seed:int=0):
        """Initializes the sampler.
        
        Args:
            dataset (DavideTrainDataset): The dataset to sample from.
            num_seqs_per_video (int): The number of sequences to sample from each clip.
            num_replicas (int, optional): The number of distributed replicas. Defaults to None.
            rank (int, optional): The rank of the current process. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.
            seed (int, optional): The random seed. Defaults to 0.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_seqs_per_video = num_seqs_per_video
        self.seed = seed

        # get clips and indices
        clips, indices = [], []
        for i, key in enumerate(dataset.keys):
            clip, _ = key.split('/')
            clips.append(clip)
            indices.append(i)
        # get unique clips (sorted)
        unique_clips = sorted(list(set(clips)))
        # get indices for each clip
        self.clip_indices = {clip: [] for clip in unique_clips}
        for i, clip in zip(indices, clips):
            self.clip_indices[clip].append(i)
        # calculate dataset length
        _dataset_len = num_seqs_per_video * len(unique_clips)
        # drop last: if the dataset is not evenly divisible, the tail of the data will be dropped
        if _dataset_len % self.num_replicas != 0:
            self.num_samples = math.ceil((_dataset_len - _dataset_len % self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(_dataset_len / self.num_replicas)
        # calculate total size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        # sample indices: for each clip, sample num_seqs_per_video indices
        indices = []
        for ids_per_clip in self.clip_indices.values():
            if len(ids_per_clip) <= self.num_seqs_per_video:
                if self.shuffle:
                    _ids_per_clip = [ids_per_clip[i] for i in torch.randperm(len(ids_per_clip), generator=g).tolist()]
                    _ids_per_clip.extend(torch.randint(min(ids_per_clip), max(ids_per_clip)+1, (self.num_seqs_per_video - len(ids_per_clip),), generator=g).tolist())
                    indices.extend(_ids_per_clip)
                else:
                    indices.extend([ids_per_clip[i % len(ids_per_clip)] for i in range(self.num_seqs_per_video)])
            else:
                if self.shuffle:
                    _ids_per_clip = [ids_per_clip[i] for i in torch.randperm(len(ids_per_clip), generator=g).tolist()[:self.num_seqs_per_video]]
                    indices.extend(_ids_per_clip)
                else:
                    indices.extend(ids_per_clip[:self.num_seqs_per_video])

        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]
        if self.shuffle:
            indices = torch.randperm(len(indices), generator=g).tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Sets the current epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.epoch = epoch