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
import numpy as np
import torch
from os import path as osp

from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir


def read_img_seq(path, datatype='gt', require_mod_crop=False, scale=1, return_imgname=False, center_crop=None):
    """
    Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        datatype (str): Data type of the images. Default is 'gt'.
                        Options: 'gt' | 'blur' | 'depth' | 'conf' | 'mono_depth_sharp' | 'mono_depth_blur'.
        require_mod_crop (bool): Require mod crop for each image. Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname (bool): Whether to return image names. Default: False.
        center_crop (dict or None): If not None, perform center cropping with the specified patch size.
                                    The dictionary should contain the key 'gt_patch_size'. Default: None.

    Returns:
        Tensor: A tensor of size (t, c, h, w), RGB, [0, 1].
        list[str]: A list of image names if return_imgname is True.
    """
    # Reading flags
    imread_flags = {
        'gt': cv2.IMREAD_COLOR,
        'blur': cv2.IMREAD_COLOR,
        'depth': cv2.IMREAD_ANYDEPTH,
        'conf': cv2.IMREAD_GRAYSCALE,
        'mono_depth_sharp': cv2.IMREAD_GRAYSCALE,
        'mono_depth_blur': cv2.IMREAD_GRAYSCALE,
    }

    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    if datatype != 'depth':
        # imgs = [cv2.imread(v, imread_flags[datatype]).astype(np.float32) / 255. for v in img_paths]
        imgs = []
        for v in img_paths:
            img = cv2.imread(v, imread_flags[datatype]).astype(np.float32) / 255.
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            imgs.append(img)
    else:
        # imgs = [cv2.imread(v, imread_flags[datatype]).astype(np.float32) / 1000. for v in img_paths]
        imgs = []
        for v in img_paths:
            img = cv2.imread(v, imread_flags[datatype]).astype(np.float32) / 1000.
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            imgs.append(img)

    if center_crop is not None:
        h, w = imgs[0].shape[:2]
        patch_size = int(center_crop['gt_patch_size']//scale) if datatype != 'gt' else int(center_crop['gt_patch_size'])
        top, left = int((h - patch_size) // 2), int((w - patch_size) // 2)
        imgs = [img[top:top + patch_size, left:left + patch_size, :] for img in imgs]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs