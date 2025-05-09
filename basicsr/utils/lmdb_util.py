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
import lmdb
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm


def make_lmdb_from_img_data(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None,
                        input_type='rgb'):
    """
    Create an LMDB database from image-like data.

    The LMDB structure is as follows:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard LMDB files. Refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt file records the meta information of the dataset.
    Each line in the file contains:
    1) image name (with extension),
    2) image shape,
    3) compression level, separated by a white space.

    For example: `ball01/00000198.png (1440,1920,3) 1` means:
    1) image name: ball01/00000198.png;
    2) image shape: (1440,1920,3);
    3) compression level: 1.

    The image name without extension is used as the LMDB key.

    If `multiprocessing_read` is True, all images are read into memory
    using multiprocessing. Ensure the server has enough memory.

    Args:
        data_path (str): Path to the directory containing images.
        lmdb_path (str): Path to save the LMDB database.
        img_path_list (list[str]): List of image paths.
        keys (list[str]): List of keys for the LMDB database.
        batch (int): Number of images to process before committing to LMDB.
            Default: 5000.
        compress_level (int): Compression level for encoding images. Default: 1.
        multiprocessing_read (bool): Use multiprocessing to read images into memory.
            Default: False.
        n_thread (int): Number of threads for multiprocessing. Default: 40.
        map_size (int | None): Map size for the LMDB environment. If None, the
            size is estimated from the images. Default: None.
        input_type (str): Type of input images. Options: 'rgb' | 'depth' | 'conf'.
            Default: 'rgb'.
    """

    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        return
    
    # read data worker. For 'rgb' and 'conf' read_img_worker, for 'depth' read_depth_worker
    read_data_worker = read_depth_worker if input_type == 'depth' else read_img_worker

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_data_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'Finish reading {len(img_path_list)} images.')

    # obtain data size for one image:
    if map_size is None:
        # reading flag. For 'rgb' and 'conf' use cv2.IMREAD_UNCHANGED, for 'depth' use cv2.IMREAD_ANYDEPTH
        imread_flag = {'rgb':cv2.IMREAD_UNCHANGED, 'depth': cv2.IMREAD_ANYDEPTH, 'conf': cv2.IMREAD_UNCHANGED}
        # read one image
        img = cv2.imread(
            osp.join(data_path, img_path_list[0]), imread_flag[input_type])
        # encode image to byte
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        # estimate map_size
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    # create lmdb environment
    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_data_worker(
                osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def read_img_worker(path, key, compress_level):
    """
    Read an image and encode it with the specified compression level.

    Args:
        path (str): Path to the image file.
        key (str): Key for the image in the LMDB database.
        compress_level (int): Compression level for encoding the image.

    Returns:
        tuple: A tuple containing:
            - str: Image key.
            - byte: Encoded image bytes.
            - tuple[int]: Image shape as (height, width, channels).
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


def read_depth_worker(path, key, compress_level):
    """
    Read a depth image and encode it with the specified compression level.

    Args:
        path (str): Path to the depth image file.
        key (str): Key for the image in the LMDB database.
        compress_level (int): Compression level for encoding the image.

    Returns:
        tuple: A tuple containing:
            - str: Image key.
            - bytes: Encoded image bytes.
            - tuple[int]: Image shape as (height, width, channels).
    """

    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth.ndim == 2:
        h, w = depth.shape
        c = 1
    else:
        h, w, c = depth.shape
    _, img_byte = cv2.imencode('.png', depth,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


class LmdbMaker():
    """
    Class for creating an LMDB database.

    Attributes:
        lmdb_path (str): Path to save the LMDB database.
        batch (int): Number of images to process before committing to LMDB.
        compress_level (int): Compression level for encoding images.
        env (lmdb.Environment): LMDB environment.
        txn (lmdb.Transaction): LMDB transaction.
        txt_file (file object): File object for the meta information file.
        counter (int): Counter for the number of images processed.
    """

    def __init__(self,
                 lmdb_path,
                 map_size=1024**4,
                 batch=5000,
                 compress_level=1):
        """
        Initialize the LmdbMaker class.

        Args:
            lmdb_path (str): Path to save the LMDB database. Must end with '.lmdb'.
            map_size (int): Map size for the LMDB environment. Default is 1TB (1024 ** 4).
            batch (int): Number of images to process before committing to LMDB. Default is 5000.
            compress_level (int): Compression level for encoding images. Default is 1.

        Raises:
            ValueError: If lmdb_path does not end with '.lmdb'.
        """
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path must end with '.lmdb'.")
        if osp.exists(lmdb_path):
            print(f'Folder {lmdb_path} already exists. Exit.')
            sys.exit(1)

        self.lmdb_path = lmdb_path
        self.batch = batch
        self.compress_level = compress_level
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
        self.counter = 0

    def put(self, img_byte, key, img_shape):
        """
        Put an image into the LMDB database.

        Args:
            img_byte (bytes): Encoded image bytes.
            key (str): Key for the image in the LMDB database.
            img_shape (tuple[int]): Shape of the image as (height, width, channels).
        """
        self.counter += 1
        key_byte = key.encode('ascii')
        self.txn.put(key_byte, img_byte)
        # write meta information
        h, w, c = img_shape
        self.txt_file.write(f'{key}.png ({h},{w},{c}) {self.compress_level}\n')
        if self.counter % self.batch == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        """
        Close the LMDB database and the meta information file.
        """
        self.txn.commit()
        self.env.close()
        self.txt_file.close()
