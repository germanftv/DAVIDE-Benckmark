# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import datetime
import logging
import time
import os
from os import path as osp

from .dist_util import get_dist_info, master_only


class MessageLogger():
    """Message logger for printing during training.

    Attributes:
        exp_name (str): Experiment name.
        interval (str): Logger interval.
        start_iter (int): Start iteration.
        max_iters (int): Total iterations.
        use_tb_logger (bool): Use tensorboard logger.
        tb_logger (obj): Tensorboard logger.
        start_time (float): Start time.
        logger (logging.Logger): Root logger.

    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        """
        Initialize the MessageLogger.

        Args:
            opt (dict): Configuration dictionary containing:
                - name (str): Experiment name.
                - logger (dict): Contains 'print_freq' (str) for logger interval.
                - train (dict): Contains 'total_iter' (int) for total iterations.
                - use_tb_logger (bool): Use tensorboard logger.
            start_iter (int, optional): Start iteration. Default is 1.
            tb_logger (obj, optional): Tensorboard logger. Default is None.
        """
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """
        Format and log a message.

        Args:
            log_vars (dict): Dictionary containing log variables:
                - epoch (int): Epoch number.
                - iter (int): Current iteration.
                - lrs (list): List of learning rates.
                - time (float, optional): Iteration time.
                - data_time (float, optional): Data loading time for each iteration.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, '
                   f'iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        
        # learning rate
        for n, v in enumerate(lrs):
            if self.use_tb_logger and 'debug' not in self.exp_name:
                self.tb_logger.add_scalar(f'lr/lr_{n}', v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    """
    Initialize a TensorBoard logger.

    Args:
        log_dir (str): Directory to save TensorBoard logs.

    Returns:
        SummaryWriter: TensorBoard SummaryWriter object for logging.
    """
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """
    Initialize a Weights and Biases (wandb) logger to sync TensorBoard logs.

    Args:
        opt (dict): Configuration dictionary containing:
            - name (str): Experiment name.
            - logger (dict): Contains 'wandb' settings:
                - project (str): Project name for wandb.
                - resume_id (str, optional): ID to resume a previous wandb run.

    Returns:
        None
    """
    import wandb
    logger = logging.getLogger('basicsr')

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        opt['logger']['wandb']['resume_id'] = wandb_id
        resume = 'never'

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt['name'],
        config=opt,
        project=project,
        sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


@master_only
def make_test_dirs(opt):
    """
    Create directories required for testing based on the provided options.

    Args:
        opt (dict): Options dictionary containing paths for various testing directories.

    """
    path_opt = opt['path'].copy()
    os.makedirs(path_opt.pop('results_root'), exist_ok=True)
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network'
                                           not in key) and ('resume'
                                                            not in key):
            os.makedirs(path, exist_ok=True)


def init_loggers(opt):
    """
    Initialize loggers for training or testing sessions.

    The function creates necessary directories in the test phase, and initializes the root logger with the specified log level and log file. 
    It logs environment information and the options dictionary, and initializes the TensorBoard and Weights and Biases (wandb) logger if specified in the options.

    Args:
        opt (dict): Options dictionary containing configuration for logging.

    Returns:
        logger (logging.Logger): Root logger session.
        tb_logger (TensorboardLogger): TensorBoard logger for the session, if enabled.

    Notes:
        - The Weights and Biases (wandb) logger is initialized before the TensorBoard logger to ensure proper synchronization.
        - The TensorBoard logger is only initialized if 'use_tb_logger' is set to True, 'debug' is not in the options name, and the phase is training.
    """
    from basicsr.utils import get_time_str

    phase = 'train' if opt.get('is_train') else 'test'

    if phase == 'test':
        # mkdir for experiments and logger
        make_test_dirs(opt)

    # initialize logger
    log_file = osp.join(opt['path']['log'],
                        f"{phase}_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
        is not None) and (opt['logger']['wandb'].get('project')
            is not None) and ('debug' not in opt['name']) and (opt['is_train']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)

    # initialize tensorboard logger
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and (opt['is_train']) and \
        'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['path'].get('experiments_root', ''), 'tb_logger', opt['name']))
    return logger, tb_logger


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
