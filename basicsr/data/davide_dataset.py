# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# - KAIR (https://github.com/cszn/KAIR.git)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------

import numpy as np
import random
import torch
import time
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms
from copy import deepcopy
import os
import glob
from os import path as osp
import shutil

from basicsr.utils import get_rank, FileClient, imfrombytes, img2tensor, get_local_rank
from basicsr.utils import video_util
from basicsr.utils import get_root_logger
from basicsr.data.transforms import (PairedAugmentation, PairedCrop, 
                            SampleToTensor, SequentialGridPatchSampler, SampleNormalization,
                            SampleResize, depth_normalization_tensor)



class DavideBaseDataset(data.Dataset):
    """Base class for loading and processing the DAVIDE dataset.

    This class provides the foundational methods for loading, processing, 
    and accessing the DAVIDE dataset.

    Attributes:
        opt (dict): Configuration dictionary for the dataset.
        logger (Logger): Logger instance for logging information.
        imglike_data (list): List storing image-like data for each data type.
        serieslike_data (list): List storing series-like data for each data type.

    """

    def __init__(self, opt):
        """Initialize the dataset.
        
        Args:
            opt (dict): Configuration dictionary for the dataset. It should contain:
                - 'dataroots' (dict): Dictionary mapping data types to their source directories.
                - 'aux_data_model' (list of str): List of auxiliary data models to be included.
                - 'local_scratch' (str, optional): Path to the local scratch directory for data transfer.
        """
        super(DavideBaseDataset, self).__init__()
        # Set the options
        self.opt = opt
        # Set the logger
        self.logger = get_root_logger()
        # Get readable data
        self.get_readable_data(opt.get('aux_data_model', ''))

    
    @staticmethod
    def transfer_to_local_scratch(opt, data_types):
        """Transfer data to local scratch storage.

        This method transfers specified data types from their source directories to a local scratch directory.
        It ensures that the data transfer is complete by comparing the sizes of the source and destination folders.
        If the transfer is successful, it logs the completion and the contents of the local scratch directory.

        Args:
            opt (dict): Configuration dictionary for the dataset. It should contain:
                - 'local_scratch' (str): Path to the local scratch directory.
                - 'dataroots' (dict): Dictionary mapping data types to their source directories.
            data_types (list of str): List of data types to be transferred.

        Example:
            >>> opt = {
            ...     'local_scratch': '/path/to/local/scratch',
            ...     'dataroots': {
            ...         'gt': '/path/to/dataset/train_gt.lmdb',
            ...         'blur': '/path/to/dataset/train_blurred.lmdb'
            ...     }
            ... }
            >>> data_types = ['gt', 'blur']
            >>> transfer_to_local_scratch(opt, data_types)
        """

        def is_folder_transferred(source_path, dest_path):
            """Check if a folder has been transferred successfully by comparing the sizes of the source and destination folders.

            This function walks through the directory tree of both the source and destination paths,
            calculates the total size of all files (excluding symbolic links), and checks if the sizes match.

            Args:
                source_path (str): The path to the source folder.
                dest_path (str): The path to the destination folder.

            Returns:
                bool: True if the folder has been transferred successfully (i.e., the total sizes match), otherwise False.

            Example:
                >>> is_folder_transferred('/path/to/source', '/path/to/destination')
                True
            """
            def get_size(start_path = '.'):
                """Calculate the total size of all files in a directory tree, excluding symbolic links.

                Args:
                    start_path (str): The starting directory path.

                Returns:
                    int: The total size of all files in bytes.
                """
                total_size = 0
                for dirpath, _, filenames in os.walk(start_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        # skip if it is symbolic link
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)

                return total_size
            
            if not os.path.exists(dest_path):
                return False
            source_size = get_size(source_path)
            dest_size = get_size(dest_path)
            return source_size == dest_size
        
        local_rank = get_local_rank()
        rank = get_rank()
        logger = get_root_logger()
        if local_rank == 0 or rank == 0:
            logger.info('Transfering data to $LOCAL_SCRATCH ...')
            # print('Transfering data to $LOCAL_SCRATCH ...')
            for datatype in data_types:
                if not os.path.exists(os.path.join(opt['local_scratch'], os.path.basename(opt['dataroots'][datatype]))):
                    shutil.copytree(opt['dataroots'][datatype], os.path.join(opt['local_scratch'], os.path.basename(opt['dataroots'][datatype])))
            
        # wait for data transfer to complete
        check_data = [is_folder_transferred(opt['dataroots'][datatype], 
                                          os.path.join(opt['local_scratch'], os.path.basename(opt['dataroots'][datatype]))
                                          ) for datatype in data_types]
        start_time = time.time()
        timeout = 60 * 60 # set a timeout of 1 hour
        # while not check_gt or not check_blur:
        while not all(check_data):
            if time.time() - start_time > timeout:
                # print("Timeout exceeded")
                logger.error("Timeout exceeded")
                break
            time.sleep(1)
            check_data = [is_folder_transferred(opt['dataroots'][datatype], 
                                              os.path.join(opt['local_scratch'], os.path.basename(opt['dataroots'][datatype]))
                                              ) for datatype in data_types]

        # check if data transfer completed successfully
        if all(check_data):
            # print("Data transfer completed successfully")
            logger.info("Data transfer completed successfully")
        # print out the contents of the local scratch directory
        for root, dirs, files in os.walk(opt['local_scratch']):
            for file in files:
                # print(os.path.join(root, file))
                logger.info(os.path.join(root, file))

    @staticmethod
    def concat_frames_in_batch_collate_fn(batch):
        """Collate function for dataloader.

        This method processes a batch of samples from the dataloader, concatenating
        frames along the batch dimension and organizing them into dictionaries for
        data and keys. It is used to prepare the data for model input.

        Args:
            batch (list): A list of samples from the dataloader. Each sample is a tuple
                containing a dictionary of data tensors and a dictionary of key values.

        Returns:
            tuple: A tuple containing two dictionaries:
                - data_batch (dict): A dictionary where keys are data types and values
                are concatenated tensors of the corresponding data type.
                - keys_batch (dict): A dictionary where keys are key types and values
                are lists of the corresponding key values.

        Example:
            >>> batch = [({'image': torch.tensor([1, 2]), 'label': torch.tensor([3])}, {'id': 1}),
                        ({'image': torch.tensor([4, 5]), 'label': torch.tensor([6])}, {'id': 2})]
            >>> data_batch, keys_batch = concat_frames_in_batch_collate_fn(batch)
            >>> print(data_batch)
            {'image': tensor([1, 2, 4, 5]), 'label': tensor([3, 6])}
            >>> print(keys_batch)
            {'id': [1, 2]}
        """
        data_batch = {}
        keys_batch = {}
        for sample, keys in batch:
            for datatype in sample:
                if datatype not in data_batch:
                    data_batch[datatype] = [sample[datatype]]
                else:
                    data_batch[datatype].append(sample[datatype])
            for keytype in keys:
                if keytype not in keys_batch:
                    keys_batch[keytype] = [keys[keytype]]
                else:
                    keys_batch[keytype].append(keys[keytype])
        
        # convert to tensor
        for datatype in data_batch:
            data_batch[datatype] = torch.cat(data_batch[datatype], dim=0)
        # for keytype in keys_batch:
        #     keys_batch[keytype] = np.array(keys_batch[keytype])

        return data_batch, keys_batch
    

    def get_readable_data(self, aux_data_model):
        """Get dataset readable data.

        This method updates the `imglike_data` and `serieslike_data` attributes
        based on the provided auxiliary data model. It categorizes the data into
        image-like and series-like data for further processing.

        Updated attributes:
            imglike_data (list of str): List of image-like data types. Initialized with
                'gt' and 'blur'. Additional types from `aux_data_model` are appended.
            serieslike_data (list of str): List of series-like data types. Initialized
                as an empty list.

        Args:
            aux_data_model (str or list of str): Auxiliary data model(s) to be included.
                It can be a single string or a list of strings. Supported values are:
                'depth', 'conf', 'mono_depth_sharp', 'mono_depth_blur'.

        Returns:
            None
        """

        if isinstance(aux_data_model, str):
            aux_data_model = [aux_data_model]
        
        self.imglike_data = ['gt', 'blur']
        self.serieslike_data = []
        if aux_data_model is None:
            return
        if len(aux_data_model) == 0:
            return
        for aux in aux_data_model:
            if aux == 'depth':
                self.imglike_data.append('depth')
            if aux == 'conf':
                self.imglike_data.append('conf')
            if aux == 'mono_depth_sharp':
                self.imglike_data.append('mono_depth_sharp')
            if aux == 'mono_depth_blur':
                self.imglike_data.append('mono_depth_blur')
        
        return



class DavideTrainDataset(DavideBaseDataset):
    """Dataset class for training on the DAVIDE dataset.

    This class is specifically designed for training on the DAVIDE dataset. It extends the DavideBaseDataset
    class and includes additional configurations and methods tailored for training purposes.
    Data is accessed through 'keys', which are generated based on the information in the 'meta_info_file'.
    
    Each line in the 'meta_info_file' contains the following information:
        - clip name
        - number of frames in the clip
        - image size (H, W, C)
        - starting frame index
        - frame interval

    If 'val_partition_file' is provided, clips not in the validation partition file are used for training.
    Meta files and validation partition files are found in the directory: 'datasets/meta_info/'.
    io_backend supports 'lmdb' and 'disk', where 'lmdb' is recommended for faster data loading during training.
    The class also supports automatic data transfer to local scratch storage for faster data access.

    Attributes:
        opt (dict): Configuration dictionary for the dataset.
        logger (Logger): Logger instance for logging information.
        imglike_data (list): List storing image-like data for each data type.
        serieslike_data (list): List storing series-like data for each data type.
        scale (int): Scale factor for the images.
        gt_size (int): Ground truth image size.
        filename_tmpl (str): Filename template.
        filename_ext (str): Filename extension.
        in_seq_length (int): Input sequence length.
        out_seq_length (int): Output sequence length.
        augmentations (dict): Augmentation configurations.
        random_reverse (bool): Whether to use random reverse augmentation.
        cropping (dict): Cropping configurations.
        file_client (FileClient): File client for IO operations.
        io_backend_opt (dict): IO backend configuration.
        is_lmdb (bool): Whether the dataset is in LMDB format.
        keys (list): List of keys for accessing the dataset.
        intervals (list): List of intervals for each clip.
        transforms (callable): Transformations to be applied to the data.

    """

    def __init__(self, opt):
        """Initialize the dataset.
        
        Args:
            opt (dict): Configuration dictionary for the training dataset. It should contain:
                - 'dataroots' (dict): Dictionary mapping data types to their source directories.
                - 'scale' (int, optional): Scale factor for the images. Default is 1.
                - 'gt_size' (int, optional): Ground truth image size. Default is 256.
                - 'filename_tmpl' (str, optional): Filename template. Default is '08d'.
                - 'filename_ext' (str, optional): Filename extension. Default is 'png'.
                - 'in_seq_length' (int): Input sequence length.
                - 'out_seq_length' (int): Output sequence length.
                - 'use_local_scratch' (bool, optional): Whether to use local scratch for data transfer. Default is False.
                - 'augmentations' (dict, optional): Augmentation configurations.
                - 'cropping' (dict, optional): Cropping configurations.
                - 'io_backend' (dict): IO backend configuration.
                - 'meta_info_file' (str): Path to the meta information file.
                - 'val_partition_file' (str, optional): Path to the validation partition file.
                - 'resize' (dict, optional): Resize configurations.
                - 'depth_normalization' (dict, optional): Depth normalization configuration. Sample configuration:
                    {
                        'type': 'seq_abs_maxnorm',
                        'depth_range': [null, null]
                    }

        """
        super(DavideTrainDataset, self).__init__(opt)

        self.scale = opt.get('scale', 1)
        self.gt_size = opt.get('gt_size', 256)
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self.filename_ext = opt.get('filename_ext', 'png')
        self.in_seq_length = opt['in_seq_length']
        self.out_seq_length = opt['out_seq_length']

        # if self.in_seq_length > 1:
        #     assert self.out_seq_length <= self.in_seq_length - 2, f"out_seq_length ({self.out_seq_length}) must be less than or equal to {self.in_seq_length - 2}"
        # else:
        #     assert self.out_seq_length == 1, f"out_seq_length ({self.out_seq_length}) must be equal to 1"

        # Check if the input sequence length is 1
        if self.in_seq_length == 1:
            assert self.out_seq_length == 1, f"out_seq_length ({self.out_seq_length}) must be equal to 1"


        # Trasfer data to local scratch if specified
        datatypes = [datatype for datatype in self.imglike_data if datatype != 'cam_blur_map']
        if opt.get('use_local_scratch', False):
            opt['local_scratch'] = os.environ.get('LOCAL_SCRATCH')
            # print(f"Local scratch path: {opt['local_scratch']}")
            self.logger.info(f"Local scratch path: {opt['local_scratch']}")
            self.transfer_to_local_scratch(opt, datatypes)
            # Set the root paths with the local scratch path
            for datatype in datatypes:
                setattr(self, f'{datatype}_root', os.path.join(opt['local_scratch'], os.path.basename(opt['dataroots'][datatype])))
        else:
            # Set the root paths with the original dataroots
            for datatype in datatypes:
                setattr(self, f'{datatype}_root', Path(opt['dataroots'][datatype]))
        
        # Augmentation configs
        self.augmentations = opt.get('augmentations', None)
        self.random_reverse = self.augmentations.pop('use_random_reverse') if self.augmentations is not None and 'use_random_reverse' in self.augmentations else False
        self.cropping = opt.get('cropping', None)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = []
            self.io_backend_opt['client_keys'] = []
            for datatype in self.imglike_data:
                self.io_backend_opt['db_paths'].append(getattr(self, f'{datatype}_root'))
                self.io_backend_opt['client_keys'].append(datatype)

        # Temporal empty key list
        self.keys = []
        self.intervals = []

        self.val_partition_file = opt.get('val_partition_file', None)
        self.resize = opt.get('resize', None)

        keys = []               # list of keys for accessing the dataset
        intervals = []          # list of interval for each clip
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                # extract clip information
                clip, num_frames, _, start_idx, interval = [int(x) if i in [1, 3, 4] else x for i, x in enumerate(line.split(' '))]
                
                # adjust the number of frames based on the input sequence length
                num_frames = num_frames - self.in_seq_length + 1

                # generate keys
                keys.extend([f'{clip}/{i:{self.filename_tmpl}}' for i in range(start_idx, start_idx+(num_frames)*interval, interval)])
                intervals.extend([interval] * (num_frames))
        
        # add keys for clips not in the validation partition file
        if self.val_partition_file is not None:
            with open(self.val_partition_file, 'r') as fin:
                val_partition = fin.read().splitlines()
            for i, v in enumerate(keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(v)
                    self.intervals.append(intervals[i])

        # self.keys = keys
        # self.intervals = intervals

        # transforms
        self.transforms = self.init_transforms()

    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            sample (dict): A dictionary containing the sample data.
            keys (dict): A dictionary containing the keys for the sample data.
        
        """
        # Initialize the FileClient reader in the first call
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # Get clip name, starting frame index, and interval
        key = self.keys[index]
        interval = self.intervals[index]
        clip_name, frame_idx = key.split('/')  # key example: bikes02/00000002

        # get neighboring frames
        end_frame_idx = int(frame_idx) + self.in_seq_length * interval
        in_neighbor_list = list(range(int(frame_idx), end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            in_neighbor_list.reverse()

        # get keys
        stencil = (self.in_seq_length - self.out_seq_length) // 2
        if stencil > 0:
            out_neighbor_list = in_neighbor_list[stencil:-stencil]
        else:
            out_neighbor_list = in_neighbor_list
        keys = {'in': [f'{clip_name}/{i:{self.filename_tmpl}}' for i in in_neighbor_list],
                'out': [f'{clip_name}/{i:{self.filename_tmpl}}' for i in out_neighbor_list]
                }

        # get the neighboring blur and GT frames
        sample = {}
        for datatype in self.imglike_data:
            imgs = []
            if datatype == 'gt' and self.in_seq_length > 1:
                stencil = (self.in_seq_length - self.out_seq_length) // 2
                if stencil > 0:
                    neighbor_list = in_neighbor_list[stencil:-stencil]
                else:
                    neighbor_list = in_neighbor_list
            else:
                neighbor_list = in_neighbor_list
            for neighbor in neighbor_list: 
                # Determine the paths
                if self.is_lmdb:
                    img_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
                else:
                    img_path= getattr(self, f'{datatype}_root') / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'    

                # Get image-like data
                img_bytes = self.file_client.get(img_path, datatype)
                try:
                    imgs.append(imfrombytes(img_bytes, datatype=datatype, float32=True))
                except Exception as e:
                    # print(f"Error reading {img_path}: {e}")
                    self.logger.error(f"Error reading {img_path}: {e}")
                    raise e
            sample[datatype] = imgs


        # transformations
        sample = self.transforms(sample)

        return sample, keys

    def __len__(self):
        """Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.keys)


    def init_transforms(self):
        """Initialize the augmentation transformations.
        
        Returns:
            callable: A callable function that applies the transformations to the data.
        """
        transforms_list = [SampleNormalization(self.imglike_data, self.serieslike_data, deepcopy(self.opt))]
        for n, transform_kwargs in enumerate([self.resize, self.cropping, self.augmentations]): 
            if n == 0:
                if transform_kwargs is not None:
                    transforms_list.append(SampleResize(self.imglike_data, **transform_kwargs))
            if n == 1:
                if transform_kwargs is not None:
                    transforms_list.append(PairedCrop(self.imglike_data, self.scale ,**transform_kwargs))
            if n == 2:
                if transform_kwargs is not None:
                    transforms_list.append(PairedAugmentation(self.imglike_data, **transform_kwargs))
        transforms_list.append(SampleToTensor(self.imglike_data, self.serieslike_data))
        return transforms.Compose(transforms_list)


class DavideValDataset(DavideBaseDataset):
    """Dataset class for validation on the DAVIDE dataset.

    This class is specifically designed for validation on the DAVIDE dataset. It extends the DavideBaseDataset
    class and includes additional configurations and methods tailored for validation purposes.
    The data is accessed by scanning the clips listed in the 'meta_info_file'.
    If 'val_partition_file' is provided, only clips in the validation partition file are used for validation.
    'val_partition_file' also provides the starting frame index for each clip.
    Meta files and validation partition files are found in the directory: 'datasets/meta_info/'.
    A total of 'n_frames_per_video' frames are used for each clip.
    The class also supports caching data in memory for faster data access.
    Caching data is recommended during validation.

    Attributes:
        opt (dict): Configuration dictionary for the dataset.
        cache_data (bool): Whether to cache data in memory.
        val_partition_file (str): Path to the validation partition file.
        n_frames_per_video (int): Number of frames per video to use.
        model_stride (int): Stride of the model.
        pad_sequence (bool): Whether to pad the sequence.
        center_crop (tuple): Center crop dimensions.
        data_info (dict): Dictionary storing paths to data.
        folders (list): List of folder names.
        img_paths_blur (dict): Dictionary storing paths to blurred images.
        img_paths_gt (dict): Dictionary storing paths to ground truth images.

    """

    def __init__(self, opt):
        """Initialize the dataset.
        
        Args:
            opt (dict): Configuration dictionary for the validation dataset. It should contain:
                - 'dataroots' (dict): Dictionary mapping data types to their source directories.
                - 'cache_data' (bool): Whether to cache data in memory.
                - 'val_partition_file' (str, optional): Path to the validation partition file.
                - 'n_frames_per_video' (int, optional): Number of frames per video to use.
                - 'model_stride' (int, optional): Stride of the model. Default is 0.
                - 'pad_sequence' (bool, optional): Whether to pad the sequence. Default is False.
                - 'center_crop' (tuple, optional): Center crop dimensions. Default is None.
                - 'meta_info_file' (str, optional): Path to the meta information file.
                - 'depth_normalization' (dict, optional): Depth normalization configuration. Default is:
                    {
                        'type': 'seq_abs_maxnorm',
                        'depth_range': [None, None]
                    }
        """
        super(DavideValDataset, self).__init__(opt)
        self.cache_data = opt['cache_data']
        self.val_partition_file = opt.get('val_partition_file', None)
        self.n_frames_per_video = opt.get('n_frames_per_video', None)
        self.model_stride = opt.get('model_stride', 0)
        self.pad_sequence = opt.get('pad_sequence', False)
        self.center_crop = opt.get('center_crop', None)

        # Initialize roots, datainfo, img_paths, and imgs
        datatypes = [datatype for datatype in self.imglike_data]
        self.data_info = {}
        for datatype in datatypes:
            setattr(self, f'{datatype}_root', Path(opt['dataroots'][datatype]))
            self.data_info[f'{datatype}_path'] = []
            setattr(self, f'img_paths_{datatype}', {})
            setattr(self, f'imgs_{datatype}', {})


        # logger = get_root_logger()

        # get subfolders
        subfolders = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                _subfolders = [line.split(' ')[0] for line in fin]
                for datatype in datatypes:
                    subfolders[datatype] = [osp.join(getattr(self, f'{datatype}_root'), key) for key in _subfolders]
        else:
            for datatype in datatypes:
                subfolders[datatype] = sorted(glob.glob(osp.join(getattr(self, f'{datatype}_root'), '*')))

        # get the starting index for each clip in the validation partition file if specified
        if self.val_partition_file is not None:
            val_partition_info = {}
            with open(self.val_partition_file, 'r') as fin:
                for lin in fin:
                    clip, start_frame = lin.split(' ')
                    val_partition_info[clip] = int(start_frame)

        # self.img_paths_blur, self.img_paths_gt = {}, {}
        
        self.folders = []
        for n, datatype in enumerate(datatypes):
            for subfolder in subfolders[datatype]:
                subfolder_name = osp.basename(subfolder)

                if self.val_partition_file is not None:
                    if subfolder_name not in val_partition_info.keys():
                        continue
                
                if n == 0:
                    self.folders.append(subfolder_name)

                # get frame list for each subfolder
                img_paths = sorted(list(video_util.scandir(subfolder, full_path=True)))

                # use subfolder index as the first item if partition file is specified
                if self.val_partition_file is not None:
                    img_paths = img_paths[val_partition_info[subfolder_name]:]

                # take the first n_frames_per_video frames if specified
                if self.n_frames_per_video is not None:
                    if self.model_stride > 0 and self.pad_sequence:
                        n_frames = self.n_frames_per_video + 2*self.model_stride
                        if len(img_paths) > n_frames:
                            img_paths = img_paths[:n_frames] if datatype != 'gt' else img_paths[self.model_stride:n_frames-self.model_stride]
                    else:
                        if len(img_paths) > self.n_frames_per_video:
                            img_paths = img_paths[:self.n_frames_per_video]

                self.data_info[f'{datatype}_path'].extend(img_paths)
 
                # Check if the data should be cached
                if self.cache_data:
                    # Data is cached

                    self.logger.info(f'Cache {datatype} data in {subfolder_name} for {self.__class__.__name__}...')
                    # Get the dictionary
                    dict_attr = getattr(self, f'img_paths_{datatype}')
                    # Set the key
                    dict_attr[subfolder_name] = img_paths
                    # Set the dictionary as the attribute again
                    setattr(self, f'img_paths_{datatype}', dict_attr)

                    # Get the dictionary
                    dict_attr = getattr(self, f'imgs_{datatype}')
                    # Set the key
                    dict_attr[subfolder_name] = video_util.read_img_seq(img_paths, datatype=datatype, center_crop=self.center_crop)
                    # Set the dictionary as the attribute again
                    setattr(self, f'imgs_{datatype}', dict_attr)
                else:
                    # Data is not cached. Paths are stored in the dictionary

                    # Get the dictionary
                    dict_attr = getattr(self, f'img_paths_{datatype}')
                    # Set the key
                    dict_attr[subfolder_name] = img_paths
                    # Set the dictionary as the attribute again
                    setattr(self, f'img_paths_{datatype}', dict_attr)

        # Find unique folder strings
        self.folders = sorted(list(set(self.folders)))

    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            sample (dict): A dictionary containing the sample data.
            keys (dict): A dictionary containing the keys for the sample data.
        """
        folder = self.folders[index]

        sample, keys = {}, {}
        keys['folder'] = folder
        for datatype in self.imglike_data:
            # Check if the data is cached
            if self.cache_data:
                # Get cached data
                imgs = getattr(self, f'imgs_{datatype}')[folder]
            else:
                # Read the images
                imgs = video_util.read_img_seq(getattr(self, f'img_paths_{datatype}')[folder], datatype=datatype, center_crop=self.center_crop)
            
            if 'depth' in datatype:
                # Normalize the depth images
                imgs = depth_normalization_tensor(imgs, **self.opt['depth_normalization'])
            
            sample[datatype] = imgs
            # Add the image paths to the keys
            keys[f'{datatype}_path'] = getattr(self, f'img_paths_{datatype}')[folder]

        return sample, keys

    def __len__(self):
        """Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.folders)
    

class DavideTestDataset(DavideBaseDataset):
    """Dataset class for validation on the DAVIDE dataset.

    This class is specifically designed for testing on the DAVIDE dataset. It extends the DavideBaseDataset
    class and includes additional configurations and methods tailored for testing purposes.
    The data is accessed by scanning the clips listed in the 'meta_info_file'.
    If 'val_partition_file' is provided, only clips in the validation partition file are used for testing.
    Meta files and validation partition files are found in the directory: 'datasets/meta_info/'.
    At each iteration, 'num_frame_testing' frames are used for testing.

    Attributes:
        opt (dict): Configuration dictionary for the dataset.
        val_partition_file (str): Path to the validation partition file.
        img_paths (list): List of image paths.
        out_img_paths (list): List of output image paths.
        in_idx_list (list): List of input indices.
        out_idx_list (list): List of output indices.
        num_frame_testing (int): Number of frames for testing per iteration.
        model_stride (int): Model stride value.
        center_crop (int): Size of the center crop.
        dataroots (dict): Dictionary mapping data types to their source directories

    """

    def __init__(self, opt):
        """Initialize the dataset.
        
        Args:
            opt (dict): Configuration dictionary for the validation dataset. It should contain:
                - 'dataroots' (dict): Dictionary mapping data types to their source directories.
                - 'num_frame_testing' (int): Number of frames for testing per iteration.
                - 'model_stride' (int, optional): Model stride value. Default is 0.
                - 'center_crop' (int, optional): Size of the center crop. Default is None.
                - 'meta_info_file' (str): Path to the meta information file.
                - 'val_partition_file' (str, optional): Path to the validation partition file.
                - 'depth_normalization' (dict, optional): Depth normalization configuration. Default is:
                    {
                        'type': 'seq_abs_maxnorm',
                        'depth_range': [None, None]
                    }
        """
        super(DavideTestDataset, self).__init__(opt)
        self.val_partition_file = opt.get('val_partition_file', None)
        self.num_frame_testing = opt.get('num_frame_testing', None)
        self.model_stride = opt.get('model_stride', 0)
        self.center_crop = opt.get('center_crop', None)
        self.dataroots = opt['dataroots']

        # get subfolders
        subfolders = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
        else:
            subfolders = sorted(glob.glob(osp.join(self.dataroots['blur'], '*')))
            subfolders = [osp.basename(subfolder) for subfolder in subfolders]

        # get the starting index for each clip in the validation partition file if specified
        if self.val_partition_file is not None:
            val_partition_info = []
            with open(self.val_partition_file, 'r') as fin:
                for lin in fin:
                    clip, _ = lin.split(' ')
                    val_partition_info.append(clip)

        self.img_paths = []
        self.out_img_paths = []
        self.in_idx_list, self.out_idx_list = [], []
        in_idx_counter, out_idx_counter = 0, 0
        for subfolder in subfolders:

            # skip the subfolder if it is not in the validation partition file
            if self.val_partition_file is not None:
                if subfolder not in val_partition_info:
                    continue

            # get frame list for each subfolder
            img_names = sorted(list(video_util.scandir(osp.join(self.dataroots['blur'], subfolder), full_path=False)))
            self.out_img_paths.extend([osp.join(subfolder, img_name) for img_name in img_names])

            # reflect paths if model_stride is greater than 0
            if self.model_stride > 0:
                pad_value = self.model_stride
                img_names = img_names[1:pad_value+1][::-1] + img_names + img_names[-pad_value-1:-1][::-1]
            
            # assign img_paths
            self.img_paths.extend([osp.join(subfolder, img_name) for img_name in img_names])

            # assign idx lists
            subfolder_idx_list = list(range(0, 
                                            len(img_names)-self.num_frame_testing-2*self.model_stride, 
                                            self.num_frame_testing)) \
                                + [max(0, len(img_names)-self.num_frame_testing-2*self.model_stride)]
            self.in_idx_list.extend([in_idx_counter + idx for idx in subfolder_idx_list])
            self.out_idx_list.extend([out_idx_counter + idx for idx in subfolder_idx_list])
            out_idx_counter += len(img_names) - 2*self.model_stride
            in_idx_counter += len(img_names)

    def __len__(self):
        """Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.in_idx_list)
    
    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            sample (dict): A dictionary containing the sample data.
            keys (dict): A dictionary containing the keys for the sample data.
        """
        in_idx = self.in_idx_list[index]
        out_idx = self.out_idx_list[index]
        in_neighbor_list = [self.img_paths[i] for i in range(in_idx, in_idx+self.num_frame_testing + 2*self.model_stride, 1)]
        out_neighbor_list = [self.img_paths[i] for i in range(in_idx+self.model_stride, in_idx+self.num_frame_testing+self.model_stride, 1)]
        
        # Add the keys
        keys = {'in': in_neighbor_list, 
                'out': out_neighbor_list, 
                'in_idx': in_idx, 
                'out_idx': out_idx}
        
        sample = {}
        for datatype in self.imglike_data:
            # Get the neighbor list
            stencil = self.model_stride
            if datatype == 'gt' and self.num_frame_testing + 2*self.model_stride > 1:
                if stencil > 0:
                    neighbor_list = in_neighbor_list[stencil:-stencil]
                else:
                    neighbor_list = in_neighbor_list
            else:
                neighbor_list = in_neighbor_list
            # Get the image paths
            img_paths = [osp.join(self.dataroots[datatype], img_path) for img_path in neighbor_list]
                
            # Read the images
            imgs = video_util.read_img_seq(img_paths, datatype=datatype, center_crop=self.center_crop)
            # Normalize the depth images
            if 'depth' in datatype:
                imgs = depth_normalization_tensor(imgs, **self.opt['depth_normalization'])
            sample[datatype] = imgs

        return sample, keys


class DavideDemoInference(DavideBaseDataset):
    """Dataset class for inference on the DAVIDE demo data.

    This class is specifically designed for inference on demo clips from the DAVIDE dataset. 
    It extends the DavideBaseDataset class and includes configurations and methods tailored 
    for processing demo clips. The data is accessed by scanning the frames of a specified clip.

    Attributes:
        opt (dict): Configuration dictionary for the dataset.
        img_paths (list): List of image paths for the input frames.
        out_img_paths (list): List of output image paths.
        in_idx_list (list): List of input indices.
        out_idx_list (list): List of output indices.
        num_frame_testing (int): Number of frames for inference per iteration.
        model_stride (int): Model stride value.
        center_crop (int): Size of the center crop.
        dataroots (dict): Dictionary mapping data types to their source directories.

    """

    def __init__(self, opt):
        """Initialize the demo dataset.
        
        Args:
            opt (dict): Configuration dictionary for the demo dataset. It should contain:
                - 'clip_name' (str): Name of the demo clip to be processed.
                - 'dataroots' (dict): Dictionary mapping data types to their source directories.
                - 'num_frame_testing' (int): Number of frames for inference per iteration.
                - 'model_stride' (int, optional): Model stride value. Default is 0.
                - 'center_crop' (int, optional): Size of the center crop. Default is None.
                - 'depth_normalization' (dict, optional): Depth normalization configuration. Default is:
                    {
                        'type': 'seq_abs_maxnorm',
                        'depth_range': [None, None]
                    }
        """
        super(DavideDemoInference, self).__init__(opt)
        self.clip_name = opt['clip_name']
        self.num_frame_testing = opt.get('num_frame_testing', None)
        self.model_stride = opt.get('model_stride', 0)
        self.center_crop = opt.get('center_crop', None)
        self.dataroots = opt['dataroots']

        self.img_paths = []
        self.out_img_paths = []
        self.in_idx_list, self.out_idx_list = [], []
        in_idx_counter, out_idx_counter = 0, 0

        # Get list of frames in the clip
        img_names = sorted(list(video_util.scandir(osp.join(self.dataroots['blur'], self.clip_name), full_path=False)))
        self.out_img_paths.extend([osp.join(self.clip_name, img_name) for img_name in img_names])

        # Reflect paths if model_stride is greater than 0
        if self.model_stride > 0:
            pad_value = self.model_stride
            img_names = img_names[1:pad_value+1][::-1] + img_names + img_names[-pad_value-1:-1][::-1]
        
        # Assign img_paths
        self.img_paths.extend([osp.join(self.clip_name, img_name) for img_name in img_names])

        # Assign idx lists
        subfolder_idx_list = list(range(0, 
                                        len(img_names)-self.num_frame_testing-2*self.model_stride, 
                                        self.num_frame_testing)) \
                            + [max(0, len(img_names)-self.num_frame_testing-2*self.model_stride)]
        self.in_idx_list.extend([in_idx_counter + idx for idx in subfolder_idx_list])
        self.out_idx_list.extend([out_idx_counter + idx for idx in subfolder_idx_list])
        out_idx_counter += len(img_names) - 2*self.model_stride
        in_idx_counter += len(img_names)

    def __len__(self):
        """Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.in_idx_list)
    
    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            sample (dict): A dictionary containing the sample data.
            keys (dict): A dictionary containing the keys for the sample data.
        """
        in_idx = self.in_idx_list[index]
        out_idx = self.out_idx_list[index]
        in_neighbor_list = [self.img_paths[i] for i in range(in_idx, in_idx+self.num_frame_testing + 2*self.model_stride, 1)]
        out_neighbor_list = [self.img_paths[i] for i in range(in_idx+self.model_stride, in_idx+self.num_frame_testing+self.model_stride, 1)]
        
        # Add the keys
        keys = {'in': in_neighbor_list, 
                'out': out_neighbor_list, 
                'in_idx': in_idx, 
                'out_idx': out_idx}
        
        sample = {}
        for datatype in self.imglike_data:
            # Get the neighbor list
            stencil = self.model_stride
            if datatype == 'gt' and self.num_frame_testing + 2*self.model_stride > 1:
                if stencil > 0:
                    neighbor_list = in_neighbor_list[stencil:-stencil]
                else:
                    neighbor_list = in_neighbor_list
            else:
                neighbor_list = in_neighbor_list
            # Get the image paths
            img_paths = [osp.join(self.dataroots[datatype], img_path) for img_path in neighbor_list]
                
            # Read the images
            imgs = video_util.read_img_seq(img_paths, datatype=datatype, center_crop=self.center_crop)
            # Normalize the depth images
            if 'depth' in datatype:
                imgs = depth_normalization_tensor(imgs, **self.opt['depth_normalization'])
            sample[datatype] = imgs

        return sample, keys
