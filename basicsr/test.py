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
import importlib
from copy import deepcopy
import torch
import os
from os import path as osp
import sys

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import (get_time_str, ProgressBar, save_config,
                           tensor2img, imwrite, parse_options, init_loggers)


# Import metrics
metric_module = importlib.import_module('basicsr.metrics')


def build_dataloaders(opt, logger):
    """
    Build dataloaders for validation or testing datasets based on the provided options.

    Args:
        opt (dict): Options dictionary containing configuration for datasets and dataloaders.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: A tuple containing the dataset and dataloader for the specified phase.

    Notes:
        - The function only supports 'val' and 'test' phases. Other phases will raise a ValueError.
        - The function assumes that the options dictionary contains the necessary configurations for datasets and dataloaders.
        - The function logs the number of images in each dataset for validation and testing purposes.
    """
    for phase, dataset_opt in opt['datasets'].items():
        if phase in ['val', 'test']:
            dataset = create_dataset(dataset_opt)
            dataloader = create_dataloader(
                dataset,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of images in {dataset_opt["name"]}: {len(dataset)}'
            )
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return dataset, dataloader


def check_pretrained(opt, logger):
    """
    Check if the pre-trained network path is specified in the options.

    Args:
        opt (dict): Options dictionary containing configuration paths.
        logger (logging.Logger): Logger for logging warnings.
    """
    if opt['path'].get('pretrain_network_g') is None:
        logger.warning('pretrain_network_g is not specified')

    

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # save options
    save_config(opt)

    # pretrained model path
    check_pretrained(opt, logger)

    # create test dataloader
    dataset, dataloader = build_dataloaders(opt, logger)

    # print("Len --- dataloader", len(dataloader))
    logger.info(f'Number of test iterations: {len(dataset)}')

    # create model
    torch.cuda.empty_cache()
    model = create_model(opt)
    model.init_testing_settings()

    # Extract test settings
    dataset_name = dataloader.dataset.opt['name']
    num_frame_testing = opt['test']['num_frame_testing']
    save_img = opt['test'].get('save_img', False)
    save_img_freq = opt['test'].get('save_img_freq', -1)
    with_metrics = opt['test'].get('metrics') is not None
    rgb2bgr = opt['test'].get('rgb2bgr', True)

    # Progress bar
    pbar = ProgressBar(len(dataloader), dataset_name)
    logger.info(f'Start testing ...')

    if with_metrics:
        import pandas as pd
        auto_csv_file = osp.join(opt['path']['csv_results'], f'{dataset_name}_metrics_autoresume.csv')
        indices_csv_file = osp.join(opt['path']['csv_results'], f'{dataset_name}_indices_autoresume.csv')
        if opt['path']['resume_state'] == 'auto' and osp.exists(auto_csv_file):
            # resume metrics
            opt_metric = deepcopy(opt['test']['metrics'])
            dtypes ={'clip_name': str, 'frame': str}
            for name, opt_ in opt_metric.items():
                dtypes[name] = float

            metrics = pd.read_csv(auto_csv_file, index_col=0, dtype=dtypes)

            # resume indices
            dtypes = {'idx': int, 'status': bool}
            indices = pd.read_csv(indices_csv_file, index_col=0, dtype=dtypes)
        else:
            # create pandas dataframe to save metrics
            columns = ['clip_name', 'frame']
            columns.extend([x for x in opt['test']['metrics'].keys()])
            img_paths = dataset.out_img_paths
            metrics = pd.DataFrame(columns=columns, index=range(len(img_paths)))

            # create pandas dataframe to save indices
            idx_list = dataset.out_idx_list
            indices = pd.DataFrame(columns=['idx', 'status'], index=range(len(idx_list)))
            indices['idx'] = idx_list
            indices['status'] = False


    # Test loop
    for i, data in enumerate(dataloader):
        data_info = data[1]['out']
        idx = data[1]['out_idx'].item()
        if opt['datasets']['test'].get('concat_frames_in_batch', False):
            data_info = data_info[0]


        # Check if metrics are already computed
        if with_metrics:
            if indices.loc[i, 'status']:
                pbar.update()
                continue
        
        # Test model
        model.feed_data(data)
        inputs = model.get_inputs()
        with torch.no_grad():
            model.output = model._test_clip(inputs, phase='test')
        visuals = model.get_current_visuals(phase='test')
        
        # Convert to numpy image
        preds = tensor2img(visuals['result'], rgb2bgr=rgb2bgr, make_grid_on=False)  
        gts = tensor2img(visuals['gt'], rgb2bgr=rgb2bgr, make_grid_on=False)
        if isinstance(preds, list) == False:
            if len(preds.shape) == 3:
                preds = [preds]
                gts = [gts]

        # Clean up gpu memory
        model.clean_tensors()
        
        # Compute metrics
        if with_metrics:
            opt_metric = deepcopy(opt['test']['metrics'])
            for name, opt_ in opt_metric.items():
                metric_type = opt_.pop('type')
                _ = opt_.pop('mode', None)
                for n , (key_info, pred, gt) in enumerate(zip(data_info, preds, gts)):
                    if opt['datasets']['test'].get('concat_frames_in_batch', False):
                        clip_name, frame_idx = key_info.split('/')
                    else:
                        clip_name, frame_idx = key_info[0].split('/')
                    metrics.loc[idx + n, 'clip_name'] = clip_name
                    metrics.loc[idx + n, 'frame'] = frame_idx
                    metrics.loc[idx + n, name] = getattr(metric_module, metric_type)(pred, gt, **opt_)
        
        # Save images
        if save_img and i % save_img_freq == 0:
            for n , (key_info, pred) in enumerate(zip(data_info, preds)):
                if opt['datasets']['test'].get('concat_frames_in_batch', False):
                    clip_name, frame_idx = key_info.split('/')
                else:
                    clip_name, frame_idx = key_info[0].split('/')
                save_preds_path = osp.join(opt['path']['visualization'], clip_name, frame_idx)
                os.makedirs(osp.dirname(save_preds_path), exist_ok=True)
                imwrite(pred, save_preds_path)
        
        # Update progress bar
        if with_metrics:
            indices.loc[i, 'status'] = True
            metrics.to_csv(auto_csv_file, index=True)
            indices.to_csv(indices_csv_file, index=True)
        pbar.update()

    pbar.close()
    logger.info(f'End of testing ...')

    # Save metrics
    if with_metrics:
        # remove previous autoresume file
        if opt['path']['resume_state'] == 'auto' and osp.exists(auto_csv_file):
            if opt['rank'] == 0:
                os.remove(auto_csv_file)
                os.remove(indices_csv_file)
        # save final metrics to csv file
        if opt['rank'] == 0:
            csv_file = osp.join(opt['path']['csv_results'], f'{dataset_name}_metrics_{get_time_str()}.csv')
            logger.info(f'Saving metrics to: {csv_file}')
            metrics.to_csv(csv_file, index=False)



if __name__ == '__main__':
    main()
