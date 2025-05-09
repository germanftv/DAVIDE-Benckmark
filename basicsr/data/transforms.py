# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import cv2
import random
import numpy as np
import torch
from basicsr.utils.img_util import img2tensor
from basicsr.utils.misc import tensor_max, tensor_min
from typing import Literal


class SampleNormalization(object):
    """
    Class that normalizes the samples in a dataset.

    Attributes:
        img_labels (list): List of image labels to be normalized.
        series_labels (list): List of series labels to be normalized.
        opt (dict): Dictionary containing normalization options.

    Methods:
        - __call__(sample): Normalizes the sample based on the provided options.
    """
    def __init__(self, img_labels, series_labels, dataset_opt) -> None:
        """
        Initializes the SampleNormalization class with image labels, series labels, and dataset options.

        Args:
            img_labels (list): List of image labels to be normalized.
            series_labels (list): List of series labels to be normalized.
            dataset_opt (dict): Dictionary containing normalization options.
        """
        super().__init__()
        self.img_labels = img_labels
        self.series_labels = series_labels
        self.opt = dataset_opt

    def __call__(self, sample):
        """
        Normalizes the sample based on the provided options.

        Args:
            sample (dict): A dictionary containing the sample data to be normalized.

        Returns:
            dict: The normalized sample.
        """
        if 'depth' in self.img_labels:
            sample['depth'] = depth_normalization(sample['depth'], **self.opt['depth_normalization'])
        return sample


# Possible depth normalization types
NORM_TYPES = Literal["frame_abs_maxnorm", "frame_log_maxnorm", "seq_abs_maxnorm", "seq_log_maxnorm", "abs_fixed-range", "log_fixed-range", "nothing"]


def depth_normalization(depth_seq, depth_range:tuple, type:NORM_TYPES='maxnorm'):
    """
    Normalizes a sequence of depth images based on the specified normalization type.

    Args:
        depth_seq (list): A list of depth images to be normalized.
        depth_range (tuple): A tuple specifying the range of depths for normalization.
        type (NORM_TYPES, optional): The type of normalization to apply. Defaults to 'maxnorm'.
            - 'frame_abs_maxnorm': Normalize each frame by its absolute maximum depth.
            - 'frame_log_maxnorm': Normalize each frame by its maximum depth in logarithmic scale.
            - 'seq_abs_maxnorm': Normalize the entire sequence by the absolute maximum depth in the sequence.
            - 'seq_log_maxnorm': Normalize the entire sequence by the maximum depth in the sequence in logarithmic scale.
            - 'abs_fixed-range': Normalize each frame by a fixed depth range.
            - 'log_fixed-range': Normalize each frame a fixed depth range in logarithmic scale.
            - 'nothing': No normalization applied.

    Returns:
        list: A list of normalized depth images.

    Raises:
        NotImplementedError: If the specified normalization type is not implemented.
    """
    if type == 'frame_abs_maxnorm':
        _depth_seq = []
        for i in range(len(depth_seq)):
            # max depth per frame
            depth_range = (np.min(depth_seq[i]), np.max(depth_seq[i]))
            depth= depth_seq[i]/depth_range[1]
            _depth_seq.append(depth)
        return _depth_seq
    elif type == 'frame_log_maxnorm':
        _depth_seq = []
        for i in range(len(depth_seq)):
            # max depth per frame
            depth_range = (0.005, np.max(depth_seq[i]))
            depth = depth_seq[i]
            depth = (np.log(depth) - np.log(depth_range[0])) / (np.log(depth_range[1]) - np.log(depth_range[0]))
            _depth_seq.append(depth)
        return _depth_seq
    elif type == 'seq_abs_maxnorm':
        # max depth per sequence
        depth_range = (np.min(np.stack(depth_seq)), np.max(np.stack(depth_seq)))
        _depth_seq = [depth/depth_range[1] for depth in depth_seq]
        return _depth_seq
    elif type == 'seq_log_maxnorm':
        # max depth per sequence
        depth_range = (0.005, np.max(np.stack(depth_seq)))
        _depth_seq = []
        for i in range(len(depth_seq)):
            depth = depth_seq[i]
            depth = (np.log(depth) - np.log(depth_range[0])) / (np.log(depth_range[1]) - np.log(depth_range[0]))
            _depth_seq.append(depth)
        return _depth_seq
    elif type == 'abs_fixed-range':
        _depth_seq = [(depth - depth_range[0]) / (depth_range[1] - depth_range[0]) for depth in depth_seq]
        return _depth_seq
    elif type == 'log_fixed-range':
        _depth_seq = []
        for i in range(len(depth_seq)):
            depth = depth_seq[i]
            depth = (np.log(depth) - np.log(depth_range[0])) / (np.log(depth_range[1]) - np.log(depth_range[0]))
            _depth_seq.append(depth)
        return _depth_seq
    elif type == 'nothing':
        return depth_seq
    else:
        return NotImplemented
    

def depth_normalization_tensor(depth_seq, depth_range:tuple, type:NORM_TYPES='maxnorm'):
    """
    Normalizes a tensor of depth images based on the specified normalization type.

    Args:
        depth_seq (torch.Tensor): A tensor of depth images to be normalized.
        depth_range (tuple): A tuple specifying the range of depths for normalization.
        type (NORM_TYPES, optional): The type of normalization to apply. Defaults to 'maxnorm'.
            - 'frame_abs_maxnorm': Normalize each frame by its absolute maximum depth.
            - 'frame_log_maxnorm': Normalize each frame by its maximum depth in logarithmic scale.
            - 'seq_abs_maxnorm': Normalize the entire sequence by the absolute maximum depth in the sequence.
            - 'seq_log_maxnorm': Normalize the entire sequence by the maximum depth in the sequence in logarithmic scale.
            - 'abs_fixed-range': Normalize each frame by a fixed depth range.
            - 'log_fixed-range': Normalize each frame a fixed depth range in logarithmic scale.
            - 'nothing': No normalization applied.

    Returns:
        torch.Tensor: A tensor of normalized depth images.

    Raises:
        NotImplementedError: If the specified normalization type is not implemented.
    """
    if type == 'frame_abs_maxnorm':
        # max depth per frame
        depth_range = (
            tensor_min(depth_seq, dims=tuple(range(1, depth_seq.dim())), keepdim=True),
            tensor_max(depth_seq, dims=tuple(range(1, depth_seq.dim())), keepdim=True)
        )
        depth_seq = depth_seq/depth_range[1]
    elif type == 'frame_log_maxnorm':
        # max depth per frame
        depth_range = (torch.tensor(0.005), 
                       tensor_max(depth_seq, dims=tuple(range(1, depth_seq.dim())), keepdim=True))
        depth_seq = (torch.log(depth_seq) - torch.log(depth_range[0])) / (torch.log(depth_range[1]) - torch.log(depth_range[0]))
    elif type == 'seq_abs_maxnorm':
        # max depth per sequence
        depth_range = (torch.min(depth_seq), torch.max(depth_seq))
        depth_seq = depth_seq/depth_range[1]
    elif type == 'seq_log_maxnorm':
        # max depth per sequence
        depth_range = (torch.tensor(0.005), torch.max(depth_seq))
        depth_seq = (torch.log(depth_seq) - torch.log(depth_range[0])) / (torch.log(depth_range[1]) - torch.log(depth_range[0]))
    elif type == 'abs_fixed-range':
        depth_seq = (depth_seq - depth_range[0]) / (depth_range[1] - depth_range[0])
    elif type == 'log_fixed-range':
        depth_seq = (torch.log(depth_seq) - torch.log(depth_range[0])) / (torch.log(depth_range[1]) - torch.log(depth_range[0]))
    elif type == 'nothing':
        pass
    else:
        return NotImplemented

    return depth_seq


class SampleToTensor(object):
    """
    SampleToTensor is a class used to convert samples in a dataset to tensors.

    Attributes:
        img_labels (list): List of image labels to be converted to tensors.
        series_labels (list): List of series labels to be converted to tensors.

    Methods:
        - __call__(sample): Converts the sample's images and series to tensors.
    """
    def __init__(self, img_labels, series_labels) -> None:
        """
        Initializes the SampleToTensor class.

        Args:
            img_labels (list): List of image labels to be converted to tensors.
            series_labels (list): List of series labels to be converted to tensors.
        """
        super().__init__()
        self.img_labels = img_labels
        self.series_labels = series_labels

    def __call__(self, sample):
        """
        Converts the sample's images and series to tensors.

        Args:
            sample (dict): A dictionary containing the sample data to be converted.

        Returns:
            dict: The sample with images and series converted to tensors.
        """
        for label in self.img_labels:
            imgs = sample[label]
            if not isinstance(imgs, list):
                imgs = [imgs]
            imgs = [img2tensor(img) for img in imgs]
            if len(imgs) == 1:
                imgs = imgs[0]
            sample[label] = imgs.unsqueeze(0) if isinstance(imgs, torch.Tensor) else torch.stack(imgs, dim=0)
        for label in self.series_labels:
            series = sample[label]
            series = torch.from_numpy(series)
            sample[label] = series
        return sample

def mod_crop(img, scale):
    """
    Crops the input image so that its dimensions are divisible by the given scale factor.

    Args:
        img (ndarray): Input image. Can be a 2D (grayscale) or 3D (color) array.
        scale (int): Scale factor by which the dimensions of the image should be divisible.

    Returns:
        ndarray: Cropped image with dimensions divisible by the scale factor.

    Raises:
        ValueError: If the input image has an unsupported number of dimensions.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


class SampleResize(object):
    """
    SampleResize is a class used to resize images in a dataset by a specified factor.

    Attributes:
        img_labels (list): List of image labels to be resized.
        factor (float): The factor by which to resize the images.

    Methods:
        - __call__(sample): Resizes the images in the sample based on the specified factor.
    """
    def __init__(self, img_labels, factor) -> None:
        """
        Initializes the SampleResize class.

        Args:
            img_labels (list): List of image labels to be resized.
            factor (float): The factor by which to resize the images.
        """
        super().__init__()
        self.img_labels = img_labels
        self.factor = factor

    def __call__(self, sample):
        """
        Resizes the images in the sample based on the specified factor.

        Args:
            sample (dict): A dictionary containing the sample data to be resized.

        Returns:
            dict: The sample with resized images.
        """
        for label in self.img_labels:
            imgs = sample[label]
            if not isinstance(imgs, list):
                imgs = [imgs]
            try:
                channels = imgs[0].shape[2]
            except IndexError:
                print(f'IndexError: {label}')
            if channels > 1:
                imgs = [cv2.resize(img, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_LINEAR) for img in imgs]
            else:
                imgs = [cv2.resize(img, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_LINEAR)[:,:,None] for img in imgs]
            if len(imgs) == 1:
                imgs = imgs[0]
            sample[label] = imgs
        
        return sample
    

class PairedCrop(object):
    """
    Class that crops paired images based on the specified type and scale.

    Attributes:
        img_labels (list): List of image labels to be cropped.
        scale (int): Scale factor for cropping.
        type (str): Type of cropping to apply. Can be 'random' or 'center'.
        gt_patch_size (int): Ground truth patch size for cropping.
        kwargs (dict): Additional keyword arguments for cropping functions.

    Methods:
        - __call__(sample): Crops the images in the sample based on the specified type and scale.
    """
    def __init__(self, img_labels, scale, type, gt_patch_size, **kwargs) -> None:
        """
        Initializes the PairedCrop class.

        Args:
            img_labels (list): List of image labels to be cropped.
            scale (int): Scale factor for cropping.
            type (str): Type of cropping to apply. Can be 'random' or 'center'.
            gt_patch_size (int): Ground truth patch size for cropping.
            kwargs (dict): Additional keyword arguments for cropping functions.
        """
        super().__init__()
        self.img_labels = img_labels
        self.gt_patch_size = gt_patch_size
        self.scale = scale
        self.type = type
        self.kwargs = kwargs

    def __call__(self, sample):
        """
        Crops the images in the sample based on the specified type and scale.

        Args:
            sample (dict): A dictionary containing the sample data to be cropped.

        Returns:
            dict: The sample with cropped images.
        """
        if self.type == 'random':
            if self.scale == 1:
                return paired_random_crop_no_scale(sample, 
                                      self.img_labels, 
                                      self.gt_patch_size)
            else:
                return paired_random_crop(sample, 
                                        self.img_labels, 
                                        self.gt_patch_size, 
                                        self.scale,
                                        **self.kwargs)
        elif self.type == 'center':
            return paired_center_crop(sample,
                                      self.img_labels,
                                      self.gt_patch_size,
                                      self.scale)


def paired_random_crop_no_scale(sample, img_labels, patch_size):
    """
    Paired random crop without scaling.

    This function crops paired images with corresponding random locations.

    Args:
        sample (dict): Sample data. Images are stored in a list for each key specified in
            img_labels. Note that all images should have the same shape. 
            If the input is an ndarray, it will be transformed to a list 
            containing itself.
        img_labels (list[str]): Image labels to be cropped.
        patch_size (int): Patch size.

    Returns:
        dict: Cropped sample data. Images are stored in a list for each key specified in
            img_labels. If returned results only have one element, just return ndarray.

    Raises:
        ValueError: If the ground truth image is smaller than the patch size.
        ValueError: If the size of the input image mismatches with the ground truth image.
    """
    # check that gt is in img_labels
    _img_labels = img_labels.copy()
    assert 'gt' in _img_labels, f'img_labels should contain gt, but got {_img_labels}'
    _img_labels.remove('gt')
    label = 'gt'

    img_gts = sample[label]
    if not isinstance(img_gts, list):
        img_gts = [img_gts]

    h_gt, w_gt, _ = img_gts[0].shape

    # randomly choose top and left coordinates for gt patch
    top = random.randint(0, h_gt - patch_size)
    left = random.randint(0, w_gt - patch_size)

    # raise error if the patch is larger than the image
    if h_gt < patch_size or w_gt < patch_size:
        raise ValueError(f'GT ({h_gt}, {w_gt}) is smaller than patch size '
                        f'({patch_size}, {patch_size}).')
    
    # crop gt patch
    img_gts = [
        v[top:top + patch_size, left:left + patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    
    # assign gt patch
    sample['gt'] = img_gts

    for label in _img_labels:
        img_inputs = sample[label]
        if not isinstance(img_inputs, list):
            img_inputs = [img_inputs]

        h_input, w_input, _ = img_inputs[0].shape

        # raise error if input size mismatches with gt size
        if h_gt != h_input or w_gt != w_input:
            raise ValueError(
                f'Size mismatches. GT ({h_gt}, {w_gt}) is not the same as '
                f'input ({h_input}, {w_input}).')

        # crop input patch
        img_inputs = [
            v[top:top + patch_size, left:left + patch_size, ...]
            for v in img_inputs]
        if len(img_inputs) == 1:
            img_inputs = img_inputs[0]

        # assign input patch
        sample[label] = img_inputs
    
    return {k: sample[k] for k in img_labels}


def paired_random_crop(sample, img_labels, gt_patch_size, scale, full_depth_mode=False):
    """
    Paired random crop.

    This function crops paired images with corresponding random locations.

    Args:
        sample (dict): Sample data. Images are stored in a list for each key specified in
            img_labels. Note that all images should have the same shape. 
            If the input is an ndarray, it will be transformed to a list 
            containing itself.
        img_labels (list[str]): Image labels to be cropped.
        gt_patch_size (int): Ground truth (GT) patch size.
        scale (int): Scale factor.
        full_depth_mode (bool, optional): If True, applies full depth mode for 'depth' and 'conf' labels.
            Defaults to False.

    Returns:
        dict: Cropped sample data. Images are stored in a list for each key specified in
            img_labels. If returned results only have one element, just return ndarray.

    Raises:
        ValueError: If the ground truth image is smaller than the patch size.
        ValueError: If the size of the input image mismatches with the ground truth image.
        ValueError: If the input image is smaller than the patch size.
    """
    # reorganize img_labels, put gt at first
    _img_labels = img_labels.copy()
    assert 'gt' in _img_labels, f'img_labels should contain gt, but got {_img_labels}'
    _img_labels.remove('gt')
    _img_labels = ['gt'] + _img_labels

    top, left = None, None

    for label in _img_labels:
        if label == 'gt':
            img_gts = sample[label]
            if not isinstance(img_gts, list):
                img_gts = [img_gts]

            h_gt, w_gt, _ = img_gts[0].shape
            crop_gt_done = False
            if h_gt < gt_patch_size or w_gt < gt_patch_size:
                raise ValueError(f'GT ({h_gt}, {w_gt}) is smaller than patch size '
                                f'({gt_patch_size}, {gt_patch_size}).')
            
        else:
            img_inputs = sample[label]
            if not isinstance(img_inputs, list):
                img_inputs = [img_inputs]

            h_input, w_input, _ = img_inputs[0].shape
            input_patch_size = gt_patch_size // scale

            if h_gt != h_input * scale or w_gt != w_input * scale:
                raise ValueError(
                    f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                    f'multiplication of "{label}" ({h_input}, {w_input}).')
            if h_input < input_patch_size or w_input < input_patch_size:
                raise ValueError(f'"{label}" ({h_input}, {w_input}) is smaller than patch size '
                                f'({input_patch_size}, {input_patch_size}).')

            # randomly choose top and left coordinates for input patch
            if top is None and left is None:
                top = random.randint(0, h_input - input_patch_size)
                left = random.randint(0, w_input - input_patch_size)

            # crop input patch
            if full_depth_mode and label in ['depth', 'conf']:
                cropped_img = []
                for img in img_inputs:
                    mask = np.zeros_like(img)
                    mask[top:top + input_patch_size, left:left + input_patch_size, :] = 1
                    img = cv2.resize(img, (input_patch_size, input_patch_size), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (input_patch_size, input_patch_size), interpolation=cv2.INTER_LINEAR)
                    img = np.stack([img, mask], axis=2)
                    cropped_img.append(img)
                img_inputs = cropped_img

            else:
                img_inputs = [
                    v[top:top + input_patch_size, left:left + input_patch_size, ...]
                    for v in img_inputs]
            if len(img_inputs) == 1:
                img_inputs = img_inputs[0]

            sample[label] = img_inputs

            # crop corresponding gt patch
            if not crop_gt_done:
                top_gt, left_gt = int(top * scale), int(left * scale)
                img_gts = [
                    v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
                    for v in img_gts
                ]
                if len(img_gts) == 1:
                    img_gts = img_gts[0]
                sample['gt'] = img_gts
                crop_gt_done = True


    return {k: sample[k] for k in img_labels}


def paired_center_crop(sample, img_labels, gt_patch_size, scale, full_depth_mode=False):
    """
    Paired center crop.

    This function crops paired images with corresponding centered locations.

    Args:
        sample (dict): Sample data. Images are stored in a list for each key specified in
            img_labels. Note that all images should have the same shape. 
            If the input is an ndarray, it will be transformed to a list 
            containing itself.
        img_labels (list[str]): Image labels to be cropped.
        gt_patch_size (int): Ground truth (GT) patch size.
        scale (int): Scale factor.
        full_depth_mode (bool, optional): If True, applies full depth mode for 'depth' and 'conf' labels.
            Defaults to False.

    Returns:
        dict: Cropped sample data. Images are stored in a list for each key specified in
            img_labels. If returned results only have one element, just return ndarray.

    Raises:
        ValueError: If the ground truth image is smaller than the patch size.
        ValueError: If the size of the input image mismatches with the ground truth image.
        ValueError: If the input image is smaller than the patch size.
    """
    # reorganize img_labels, put gt at first
    _img_labels = img_labels.copy()
    assert 'gt' in _img_labels, f'img_labels should contain gt, but got {_img_labels}'
    _img_labels.remove('gt')
    _img_labels = ['gt'] + _img_labels

    top, left = None, None
    for label in _img_labels:
        if label == 'gt':
            img_gts = sample[label]
            if not isinstance(img_gts, list):
                img_gts = [img_gts]

            h_gt, w_gt, _ = img_gts[0].shape
            crop_gt_done = False
            if h_gt < gt_patch_size or w_gt < gt_patch_size:
                raise ValueError(f'GT ({h_gt}, {w_gt}) is smaller than patch size '
                                f'({gt_patch_size}, {gt_patch_size}).')
            
        else:
            img_inputs = sample[label]
            if not isinstance(img_inputs, list):
                img_inputs = [img_inputs]

            h_input, w_input, _ = img_inputs[0].shape
            input_patch_size = gt_patch_size // scale

            if h_gt != h_input * scale or w_gt != w_input * scale:
                raise ValueError(
                    f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                    f'multiplication of input ({h_input}, {w_input}).')
            if h_input < input_patch_size or w_input < input_patch_size:
                raise ValueError(f'input ({h_input}, {w_input}) is smaller than patch size '
                                f'({input_patch_size}, {input_patch_size}).')

            # Choose top and left coordinates for input patch to be centered
            if top is None and left is None:
                top = (h_input - input_patch_size) // 2
                left = (w_input - input_patch_size) // 2

            # crop input patch
            if full_depth_mode and label in ['depth', 'conf']:
                cropped_img = []
                for img in img_inputs:
                    mask = np.zeros_like(img)
                    mask[top:top + input_patch_size, left:left + input_patch_size, :] = 1
                    img = cv2.resize(img, (input_patch_size, input_patch_size), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (input_patch_size, input_patch_size), interpolation=cv2.INTER_LINEAR)
                    img = np.stack([img, mask], axis=2)
                    cropped_img.append(img)
                img_inputs = cropped_img

            else:
                img_inputs = [
                    v[top:top + input_patch_size, left:left + input_patch_size, ...]
                    for v in img_inputs]
            if len(img_inputs) == 1:
                img_inputs = img_inputs[0]

            sample[label] = img_inputs

            # crop corresponding gt patch
            if not crop_gt_done:
                top_gt, left_gt = (h_gt - gt_patch_size) // 2, (w_gt - gt_patch_size) // 2
                img_gts = [
                    v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
                    for v in img_gts
                ]
                if len(img_gts) == 1:
                    img_gts = img_gts[0]
                sample['gt'] = img_gts
                crop_gt_done = True


    return {k: sample[k] for k in img_labels}


class SequentialGridPatchSampler(object):
    """
    SequentialGridPatchSampler is a class used to sample patches from images in a sequential grid manner.

    Attributes:
        img_labels (list): List of image labels to be sampled.
        gt_patch_size (int): Ground truth (GT) patch size.
        scale (int): Scale factor.
        num_patches (int): Total number of patches per side.
        patch_idx (int): Index of the current patch.
        kwargs (dict): Additional keyword arguments for sampling functions.

    Methods:
        - __call__(sample): Samples patches from the images in the sample based on the specified grid and scale.
    """
    def __init__(self, img_labels, gt_patch_size, scale, num_patches_per_side, patch_idx, **kwargs) -> None:
        """
        Initializes the SequentialGridPatchSampler class with image labels, GT patch size, scale, number of patches per side, and patch index.

        Args:
            img_labels (list): List of image labels to be sampled.
            gt_patch_size (int): Ground truth (GT) patch size.
            scale (int): Scale factor.
            num_patches_per_side (int): Number of patches per side.
            patch_idx (int): Index of the current patch.
            kwargs (dict): Additional keyword arguments for sampling functions.
        """
        super().__init__()
        self.img_labels = img_labels
        self.gt_patch_size = gt_patch_size
        self.scale = scale
        self.num_patches = int(num_patches_per_side ** 2)
        self.patch_idx = patch_idx
        self.kwargs = kwargs

    def __call__(self, sample):
        """
        Samples patches from the images in the sample based on the specified grid and scale.

        Args:
            sample (dict): A dictionary containing the sample data to be cropped and sampled.

        Returns:
            dict: The sample with cropped and sampled patches.
        """
        sample = paired_center_crop(sample,
                                    self.img_labels,
                                    self.gt_patch_size * int(np.sqrt(self.num_patches)),
                                    self.scale)
        sample = paired_grid_patch_sampler(sample,
                                            self.img_labels,
                                            self.gt_patch_size,
                                            self.scale,
                                            self.patch_idx,
                                            **self.kwargs)
        self.patch_idx += 1
        if self.patch_idx == self.num_patches:
            self.patch_idx = 0
        return sample



def paired_grid_patch_sampler(sample, img_labels, gt_patch_size, scale, patch_idx, full_depth_mode=False):
    """
    Sample paired patches from a grid index.

    This function crops paired images with corresponding grid locations based on the patch index.

    Args:
        sample (dict): Sample data. Images are stored in a list for each key specified in
            img_labels. Note that all images should have the same shape. 
            If the input is an ndarray, it will be transformed to a list 
            containing itself.
        img_labels (list[str]): Image labels to be cropped.
        gt_patch_size (int): Ground truth (GT) patch size.
        scale (int): Scale factor.
        patch_idx (int): Index of the patch to be sampled from the grid.
        full_depth_mode (bool, optional): If True, applies full depth mode for 'depth' and 'conf' labels.
            Defaults to False.

    Returns:
        dict: Cropped sample data. Images are stored in a list for each key specified in
            img_labels. If returned results only have one element, just return ndarray.

    Raises:
        ValueError: If the ground truth image is smaller than the patch size.
        ValueError: If the size of the input image mismatches with the ground truth image.
        ValueError: If the input image is smaller than the patch size.
    """
    # reorganize img_labels, put gt at first
    _img_labels = img_labels.copy()
    assert 'gt' in _img_labels, f'img_labels should contain gt, but got {_img_labels}'
    _img_labels.remove('gt')
    _img_labels = ['gt'] + _img_labels

    top, left = None, None

    for label in _img_labels:
        if label == 'gt':
            img_gts = sample[label]
            if not isinstance(img_gts, list):
                img_gts = [img_gts]

            h_gt, w_gt, _ = img_gts[0].shape
            crop_gt_done = False
            if h_gt < gt_patch_size or w_gt < gt_patch_size:
                raise ValueError(f'GT ({h_gt}, {w_gt}) is smaller than patch size '
                                f'({gt_patch_size}, {gt_patch_size}).')
            num_patches_per_h_gt = h_gt // gt_patch_size
            num_patches_per_w_gt = w_gt // gt_patch_size
            
        else:
            img_inputs = sample[label]
            if not isinstance(img_inputs, list):
                img_inputs = [img_inputs]

            h_input, w_input, _ = img_inputs[0].shape
            input_patch_size = gt_patch_size // scale

            if h_gt != h_input * scale or w_gt != w_input * scale:
                raise ValueError(
                    f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                    f'multiplication of "{label}" ({h_input}, {w_input}).')
            if h_input < input_patch_size or w_input < input_patch_size:
                raise ValueError(f'"{label}" ({h_input}, {w_input}) is smaller than patch size '
                                f'({input_patch_size}, {input_patch_size}).')
            
            num_patches_per_h_input = h_input // input_patch_size
            num_patches_per_w_input = w_input // input_patch_size

            # choose top and left coordinates for input patch based on patch_idx
            if top is None and left is None:
                i, j = patch_idx // num_patches_per_w_input, patch_idx % num_patches_per_w_input
                top, left = i * input_patch_size, j * input_patch_size

            # crop input patch
            if full_depth_mode and label in ['depth', 'conf']:
                cropped_img = []
                for img in img_inputs:
                    mask = np.zeros_like(img)
                    mask[top:top + input_patch_size, left:left + input_patch_size, :] = 1
                    img = cv2.resize(img, (input_patch_size, input_patch_size), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (input_patch_size, input_patch_size), interpolation=cv2.INTER_LINEAR)
                    img = np.stack([img, mask], axis=2)
                    cropped_img.append(img)
                img_inputs = cropped_img

            else:
                img_inputs = [
                    v[top:top + input_patch_size, left:left + input_patch_size, ...]
                    for v in img_inputs]
            if len(img_inputs) == 1:
                img_inputs = img_inputs[0]

            sample[label] = img_inputs

            # crop corresponding gt patch
            if not crop_gt_done:
                i, j = patch_idx // num_patches_per_w_gt, patch_idx % num_patches_per_w_gt
                top_gt, left_gt = i * gt_patch_size, j * gt_patch_size
                img_gts = [
                    v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
                    for v in img_gts
                ]
                if len(img_gts) == 1:
                    img_gts = img_gts[0]
                sample['gt'] = img_gts
                crop_gt_done = True


    return {k: sample[k] for k in img_labels}

# def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path):
#     """Paired random crop.

#     It crops lists of lq and gt images with corresponding locations.

#     Args:
#         img_gts (list[ndarray] | ndarray): GT images. Note that all images
#             should have the same shape. If the input is an ndarray, it will
#             be transformed to a list containing itself.
#         img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
#             should have the same shape. If the input is an ndarray, it will
#             be transformed to a list containing itself.
#         gt_patch_size (int): GT patch size.
#         scale (int): Scale factor.
#         gt_path (str): Path to ground-truth.

#     Returns:
#         list[ndarray] | ndarray: GT images and LQ images. If returned results
#             only have one element, just return ndarray.
#     """

#     if not isinstance(img_gts, list):
#         img_gts = [img_gts]
#     if not isinstance(img_lqs, list):
#         img_lqs = [img_lqs]

#     h_lq, w_lq, _ = img_lqs[0].shape
#     h_gt, w_gt, _ = img_gts[0].shape
#     lq_patch_size = gt_patch_size // scale

#     if h_gt != h_lq * scale or w_gt != w_lq * scale:
#         raise ValueError(
#             f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
#             f'multiplication of LQ ({h_lq}, {w_lq}).')
#     if h_lq < lq_patch_size or w_lq < lq_patch_size:
#         raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
#                          f'({lq_patch_size}, {lq_patch_size}). '
#                          f'Please remove {gt_path}.')

#     # randomly choose top and left coordinates for lq patch
#     top = random.randint(0, h_lq - lq_patch_size)
#     left = random.randint(0, w_lq - lq_patch_size)

#     # crop lq patch
#     img_lqs = [
#         v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
#         for v in img_lqs
#     ]

#     # crop corresponding gt patch
#     top_gt, left_gt = int(top * scale), int(left * scale)
#     img_gts = [
#         v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
#         for v in img_gts
#     ]
#     if len(img_gts) == 1:
#         img_gts = img_gts[0]
#     if len(img_lqs) == 1:
#         img_lqs = img_lqs[0]
#     return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """
    Augment images with horizontal flips and rotations (0, 90, 180, 270 degrees).

    This function applies horizontal flips and rotations to the input images and flows.
    Vertical flips and transpositions are used for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool, optional): Apply horizontal flip. Default is True.
        rotation (bool, optional): Apply rotation. Default is True.
        flows (list[ndarray], optional): Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list. Dimension is (h, w, 2). Default is None.
        return_status (bool, optional): Return the status of flip and rotation. Default is False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
        tuple: If return_status is True, returns a tuple containing the augmented images and a tuple
            of booleans indicating the status of (hflip, vflip, rot90).

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs

class PairedAugmentation(object):
    """
    PairedAugmentation is a class used to apply paired augmentations to images.

    This class applies horizontal flips and rotations to paired images based on the specified labels.

    Attributes:
        img_labels (list[str]): List of image labels to be augmented.
        hflip (bool): Whether to apply horizontal flips. Default is True.
        rotation (bool): Whether to apply rotations (0, 90, 180, 270 degrees). Default is True.

    Methods:
        - __call__(sample): Applies the specified augmentations to the images in the sample.
    """
    
    def __init__(self, img_labels, use_flip=True, use_rot=True) -> None:
        """
        Initializes the PairedAugmentation class.

        Args:
            img_labels (list[str]): List of image labels to be augmented.
            use_flip (bool, optional): Whether to apply horizontal flips. Default is True.
            use_rot (bool, optional): Whether to apply rotations (0, 90, 180, 270 degrees). Default is True.
        """
        super().__init__()
        self.img_labels = img_labels
        self.hflip = use_flip
        self.rotation = use_rot

    def __call__(self, sample):
        """
        Applies the specified augmentations to the images in the sample.

        Args:
            sample (dict): A dictionary containing the sample data to be augmented.

        Returns:
            dict: The sample with augmented images.
        """
        return paired_augmentation(sample, 
                                    self.img_labels, 
                                    self.hflip, 
                                    self.rotation, 
                                    return_status=False)


def paired_augmentation(sample, img_labels, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    for label in img_labels:
        is_flow = True if label == 'flow' else False
        imgs = sample[label]
        if not isinstance(imgs, list):
            imgs = [imgs]

        augment_fc = _augment_flow if is_flow else _augment
        imgs = [augment_fc(img, hflip, vflip, rot90) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]
        sample[label] = imgs

    if return_status:
        return sample, (hflip, vflip, rot90)
    else:
        return sample


def _augment(img, hflip, vflip, rot90):
    """
    Apply augmentations to an image including horizontal flip, vertical flip, and 90-degree rotation.

    This function applies the specified augmentations to the input image. The augmentations include
    horizontal flip, vertical flip, and 90-degree rotation.

    Args:
        img (ndarray): The input image to be augmented.
        hflip (bool): Whether to apply a horizontal flip.
        vflip (bool): Whether to apply a vertical flip.
        rot90 (bool): Whether to apply a 90-degree rotation.

    Returns:
        ndarray: The augmented image.
    """
    if hflip:  # horizontal
        img = np.flip(img, axis=1).copy()
    if vflip:  # vertical
        img = np.flip(img, axis=0).copy()
    if rot90:
        img = img.transpose(1, 0, 2)
    return img


def _augment_flow(flow, hflip, vflip, rot90):
    """
    Apply augmentations to flow data including horizontal flip, vertical flip, and 90-degree rotation.

    This function applies the specified augmentations to the input flow data. The augmentations include
    horizontal flip, vertical flip, and 90-degree rotation. The flow data is expected to have the shape (h, w, 2),
    where the last dimension represents the flow vectors.

    Args:
        flow (ndarray): The input flow data to be augmented. Shape should be (h, w, 2).
        hflip (bool): Whether to apply a horizontal flip.
        vflip (bool): Whether to apply a vertical flip.
        rot90 (bool): Whether to apply a 90-degree rotation.

    Returns:
        ndarray: The augmented flow data.
    """
    if hflip:  # horizontal
        flow = np.flip(flow, axis=1).copy()
        flow[:, :, 0] *= -1  # Invert the horizontal component
    if vflip:  # vertical
        flow = np.flip(flow, axis=0).copy()
        flow[:, :, 1] *= -1  # Invert the vertical component
    if rot90:
        flow = flow.transpose(1, 0, 2)
        flow = flow[:, :, [1, 0]]  # Swap the flow components
        flow[:, :, 1] *= -1  # Invert the new vertical component (original horizontal component)
    return flow


# def img_rotate(img, angle, center=None, scale=1.0):
#     """Rotate image.

#     Args:
#         img (ndarray): Image to be rotated.
#         angle (float): Rotation angle in degrees. Positive values mean
#             counter-clockwise rotation.
#         center (tuple[int]): Rotation center. If the center is None,
#             initialize it as the center of the image. Default: None.
#         scale (float): Isotropic scale factor. Default: 1.0.
#     """
#     (h, w) = img.shape[:2]

#     if center is None:
#         center = (w // 2, h // 2)

#     matrix = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated_img = cv2.warpAffine(img, matrix, (w, h))
#     return rotated_img

# def data_augmentation(image, mode):
#     """
#     Performs data augmentation of the input image
#     Input:
#         image: a cv2 (OpenCV) image
#         mode: int. Choice of transformation to apply to the image
#                 0 - no transformation
#                 1 - flip up and down
#                 2 - rotate counterwise 90 degree
#                 3 - rotate 90 degree and flip up and down
#                 4 - rotate 180 degree
#                 5 - rotate 180 degree and flip
#                 6 - rotate 270 degree
#                 7 - rotate 270 degree and flip
#     """
#     if mode == 0:
#         # original
#         out = image
#     elif mode == 1:
#         # flip up and down
#         out = np.flipud(image)
#     elif mode == 2:
#         # rotate counterwise 90 degree
#         out = np.rot90(image)
#     elif mode == 3:
#         # rotate 90 degree and flip up and down
#         out = np.rot90(image)
#         out = np.flipud(out)
#     elif mode == 4:
#         # rotate 180 degree
#         out = np.rot90(image, k=2)
#     elif mode == 5:
#         # rotate 180 degree and flip
#         out = np.rot90(image, k=2)
#         out = np.flipud(out)
#     elif mode == 6:
#         # rotate 270 degree
#         out = np.rot90(image, k=3)
#     elif mode == 7:
#         # rotate 270 degree and flip
#         out = np.rot90(image, k=3)
#         out = np.flipud(out)
#     else:
#         raise Exception('Invalid choice of image transformation')

#     return out

# def random_augmentation(*args):
#     out = []
#     if random.randint(0,1) == 1:
#         flag_aug = random.randint(1,7)
#         for data in args:
#             out.append(data_augmentation(data, flag_aug).copy())
#     else:
#         for data in args:
#             out.append(data)
#     return out