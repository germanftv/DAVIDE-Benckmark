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
import datetime
import math
import time
import torch
import os
import re
import sys
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import BalancedClipSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, make_exp_dirs, mkdir_and_rename,
                           parse_options, update_options, init_loggers)


def build_dataloaders(opt, logger):
    """
    Build dataloaders for training and validation datasets based on the provided options.

    Args:
        opt (dict): Options dictionary containing configuration for datasets, training, and distributed settings.
        logger (logging.Logger): Logger for logging training and validation details.

    Returns:
        tuple: A tuple containing the training dataloader, training sampler, validation dataloader, total epochs, and total iterations.

    The function performs the following steps:
        1. Iterates over the datasets specified in the options.
        2. For the 'train' phase:
            - Creates the training dataset.
            - Creates a balanced clip sampler for the training dataset.
            - Creates the training dataloader using the dataset and sampler.
            - Logs training statistics including the number of sequences, samples per epoch, batch size, world size, iterations per epoch, total epochs, and total iterations.
        3. For the 'val' phase:
            - Creates the validation dataset.
            - Creates the validation dataloader.
            - Logs validation statistics including the number of validation images or folders.
        4. Raises a ValueError if an unrecognized dataset phase is encountered.

    Notes:
        - The function calculates the total number of iterations and epochs based on the provided options.
        - The function ensures that either 'total_epochs' or 'total_iter' is specified in the training options.
    """

    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # create train dataset
            train_set = create_dataset(dataset_opt)
            # create train sampler
            train_sampler = BalancedClipSampler(train_set,
                                                dataset_opt['num_seqs_per_video'],
                                                num_replicas=opt['world_size'],
                                                rank=opt['rank'],
                                                shuffle=dataset_opt.get('use_shuffle', True),
                                                seed=opt['manual_seed'])
            # create train dataloader
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            # log traning details
            num_iter_per_epoch = math.ceil(
                train_sampler.total_size / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            if opt['train'].get('total_epochs') is not None:
                total_epochs = opt['train']['total_epochs']
                total_iters = total_epochs * num_iter_per_epoch
                opt['train']['total_iter'] = total_iters
            elif opt['train'].get('total_iter') is not None:
                total_iters = int(opt['train']['total_iter'])
                total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
                opt['train']['total_epochs'] = total_epochs
            else:
                raise ValueError(
                    'Please specify "total_epochs" or "total_iter" in training phase.')
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train sequences: {len(train_set)}'
                f'\n\tNumber of samples per epoch: {train_sampler.num_samples}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase == 'val':
            # create validation dataset
            val_set = create_dataset(dataset_opt)
            # create validation dataloader
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            
            # log validation details
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}'
            )
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def get_resume_state(opt):
    """
    Determine the resume state for training based on the provided options.

    Args:
        opt (dict): Options dictionary containing configuration for paths, logging, and training states.

    Returns:
        tuple: A tuple containing the resume state (if any) and the updated options dictionary.

    The function performs the following steps:
        1. Checks if automatic resume is enabled by examining the 'resume_state' option.
        2. If 'resume_state' is set to 'auto':
            - Lists the state files in the training states directory.
            - Filters out state files with negative numbers in their names.
            - Finds the latest state file based on the highest iteration number.
            - Updates the 'resume_state' option with the path to the latest state file.
            - Sets the pre-trained model path based on the latest state file if not already specified.
        3. If 'resume_state' is None, sets the resume state path to None.
        4. If 'resume_state' is specified, uses the provided state file path.
        5. Loads the resume state from the specified path if it exists.
        6. Loads the Weights and Biases (wandb) ID from the resume state if it exists and updates the options.
        7. Creates necessary directories for experiments and logging if no resume state is found.
        8. Ensures the TensorBoard logger directory is created and renamed if necessary.

    Notes:
        - The function handles exceptions when listing state files and ensures that invalid state files are ignored.
        - The function ensures that the TensorBoard logger directory is only created if 'use_tb_logger' is enabled, the name does not contain 'debug', and the rank is 0.
    """

    # automatic resume ...
    if opt['path']['resume_state'] == 'auto':
        state_folder_path = opt['path']['training_states']
        try:
            states = os.listdir(state_folder_path)
            # Remove states with negative numbers in their names
            states = [state for state in states if not re.search(r"-\d+.state", state)]
        except:
            states = []
        
        # Find the latest state file
        if len(states) > 0:
            iter_exist = []
            for file_ in states:
                iter_current = re.findall(r"(\d+).state", file_)
                iter_exist.append(int(iter_current[0]))
            max_iter = max(iter_exist)
            max_state_file = '{}.state'.format(max_iter)
            resume_state_path = os.path.join(state_folder_path, max_state_file)
            opt['path']['resume_state'] = resume_state_path

            # define pretrained model path based on lastest state file
            if opt['path'].get('pretrain_network_g') is None:
                opt['path']['pretrain_network_g'] = osp.join(
                    opt['path']['models'],
                    'net_g_{}.pth'.format(max_iter))
        else:
            resume_state_path = None
    # not resume ...
    elif opt['path']['resume_state'] is None:
        resume_state_path = None
    # resume from specified state file ...
    else:
        resume_state_path = opt['path']['resume_state']

    # load resume states if necessary
    if resume_state_path is not None:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'])
    else:
        resume_state = None

    # load wandb id if exist
    if resume_state is not None and 'wandb_id' in resume_state:
        opt['logger']['wandb']['resume_id'] = resume_state['wandb_id']

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['path'].get('experiments_root', ''), 'tb_logger', opt['name']))
    
    return resume_state, opt

    

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # resume state
    resume_state, opt = get_resume_state(opt)

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = build_dataloaders(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # update options
    opt = update_options(opt, total_epochs=total_epochs, total_iters=total_iters)

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.init_training_settings()
        model.resume_training(resume_state)  # handle optimizers and scheduler
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        del resume_state
        torch.cuda.empty_cache()
    else:
        model = create_model(opt)
        model.init_training_settings()
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # Starting epoch and iteration
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # --------------------- training loop -------------------------

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if opt['rank'] == 0 and current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if opt['rank'] == 0 and current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                # print("saving")
                model.save(epoch, current_iter)

            # validation every val_freq iterations
            if opt['rank'] == 0 and opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image, epoch)
 
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
            # end of iter
        epoch += 1
        # end of epoch
    # end of training

    if tb_logger:
        # Add the best validation metrics to tensorboard
        metric_dict = {f'metric/{model.best_metric_name}': model.best_metric_value,
                        'metric/flops': model.flops,
                        'metric/params': model.total_params}
        tb_logger.add_hparams(
            {}, metric_dict)

    # consuming time
    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    # save the latest model
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
