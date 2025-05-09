# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import argparse
import yaml
import random
from collections import OrderedDict
import os
from os import path as osp
from omegaconf import OmegaConf, DictConfig

from basicsr.utils.misc import get_time_str, set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only


@master_only
def save_config(opt):
    """Save config to the disk.

    Args:
        opt (dict): Options.
    """
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    dump_path = osp.join(opt_path_copy, 'config_'+get_time_str()+'.yml')
    with open(dump_path, 'w') as f:
        _, Dumper = ordered_yaml()
        yaml.dump(opt, f, indent=2, Dumper=Dumper)


def ordered_yaml():
    """
    Support OrderedDict for YAML serialization and deserialization.

    This function modifies the YAML Loader and Dumper to support `OrderedDict`,
    ensuring that the order of dictionary keys is preserved when loading and dumping YAML files.

    Returns:
        tuple: A tuple containing the modified YAML Loader and Dumper classes.

    The function attempts to use the C-based Loader and Dumper for better performance.
    If they are not available, it falls back to the pure Python implementations.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse_args_demo():
    """
    Parse command line arguments for demo script.

    Returns:
        dict: Parsed and updated options dictionary.

    This function sets up the command line argument parser for demo mode, allowing users to specify paths to input images,
    output directories, and other options related to the demo process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clip',
        type=str,
        required=True,
        choices=["toy01", "farm03", "indoors02", "play_ground05", "robot04", "birds07"],
        help='Name of the video clip to be processed')
    parser.add_argument(
        '--exp',
        type=str,
        required=True,
        choices=['depth_quality', 'depth_impact', 'sota_comparison'],
        help='Name of the experiment.')
    parser.add_argument(
        '--results_dir',
        type=str,
        default=os.getenv('DEMO_RESULTS_ROOT', './results/demo-data'),
        help='Directory to save results. Defaults to the value of the $DEMO_RESULTS_ROOT environment variable, or "./results" if not set.')
    parser.add_argument(
        '--model',
        type=str,
        default='Shiftnet_RGBD',
        choices=['DGN_single_frame', 'DGN_multi_frame', 'EDVR', 'VRT', 'RVRT', 'Shiftnet_baseRGB', 'Shiftnet_RGBD'],
        help='Model to be used for processing. Choices are: DGN_single_frame, DGN_multi_frame, EDVR, VRT, RVRT, Shiftnet_baseRGB, Shiftnet_RGBD.')
    parser.add_argument(
        '--depth',
        type=str,
        default=None,
        choices=['sensor', 'mono_blur', 'mono_sharp', None],
        help='Depth type to be used. Choices are: sensor, mono_blur, mono_sharp, None. Default is None.')
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='Window size for processing. Default is None.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed used for pre-training. Default is None.')
    parser.add_argument(
        '--configs_dir',
        type=str,
        default='./options/demo',
        help='Directory containing the configuration files. Default is "./options/demo".')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='./model_zoo/davide_checkpoints',
        help='Directory containing the pre-trained model checkpoints. Default is "./model_zoo/davide_checkpoints".')
    args = parser.parse_args()

    # Read config options
    exp_folders = {
        'depth_quality': 'E0_depth_quality',
        'depth_impact': 'E1_depth_impact',
        'sota_comparison': 'E2_SOTA_comparison'
    }
    if args.exp == 'depth_quality':
        # Checks for depth_quality experiment
        assert args.depth is not None, 'Depth type must be specified for depth_quality experiment.'
        assert args.model == 'Shiftnet_RGBD', 'Model must be Shiftnet_RGBD for depth_quality experiment.'
        assert args.window_size is None, 'Window size must be None.'
        assert args.seed is None, 'Seed must be None.'

        # Set configurations
        default_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'defaults.yml')
        arch_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'archs', f'{args.model}.yml')
        depth_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'depth_opts', f'{args.depth}.yml')
        configs = [default_config, arch_config, depth_config]

        # Load options
        opt = load_yaml_options(configs)

        # Set additional options
        opt['name'] = f'{args.exp}_{args.clip}_{args.depth}'
        opt['path']['results_root'] = os.path.join(args.results_dir, opt['path']['results_root'])
        opt['datasets']['test']['clip_name'] = args.clip

        ckpt_files = {
            'sensor': 'Shiftnet_sensor_depth.pth',
            'mono_blur': 'Shiftnet_mono_depth_blur.pth',
            'mono_sharp': 'Shiftnet_mono_depth_sharp.pth'
        }
        opt['path']['pretrain_network_g'] = os.path.join(args.ckpt_dir, exp_folders[args.exp], ckpt_files[args.depth])

    if args.exp == 'depth_impact':
        # Checks for depth_impact experiment
        assert args.depth is None, 'Depth type must be None for depth_impact experiment. This is adjusted automatically based on the model.'
        assert args.model in ['Shiftnet_baseRGB', 'Shiftnet_RGBD'], 'Model must be Shiftnet_baseRGB or Shiftnet_RGBD for depth_impact experiment.'
        assert args.window_size is not None, 'Window size must be specified.'

        if args.seed is None:
            print('By default, using pretrained models with seed 13.')
            args.seed = 13

        # Set configurations
        default_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'defaults.yml')
        arch_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'archs', f'{args.model}.yml')
        configs = [default_config, arch_config]

        # Load options
        opt = load_yaml_options(configs)

        # Set additional options
        opt['aux_data_model'] = ['depth'] if args.model == 'Shiftnet_RGBD' else ['none']
        opt['in_seq_length'] = args.window_size
        opt['out_seq_length'] = args.window_size - 2 if args.window_size > 1 else 1
        opt['manual_seed'] = args.seed
        opt['name'] = f'{args.exp}_{args.clip}_{args.model}_T{args.window_size:02d}_seed{args.seed}'
        opt['path']['results_root'] = os.path.join(args.results_dir, opt['path']['results_root'])
        opt['datasets']['test']['clip_name'] = args.clip
        if args.window_size == 1:
            opt['datasets']['test']['num_frame_testing'] = 1


        ckpt_start_tmp = {
            'Shiftnet_baseRGB': 'baseRGB_Shiftnet',
            'Shiftnet_RGBD': 'RGBD_Shiftnet'
        }

        ckpt_file = f'{ckpt_start_tmp[args.model]}_T{args.window_size:02d}_seed{args.seed}.pth'
        opt['path']['pretrain_network_g'] = os.path.join(args.ckpt_dir, exp_folders[args.exp], ckpt_file)
    
    if args.exp == 'sota_comparison':
        # Checks for sota_comparison experiment
        assert args.depth is None, 'Depth type must be None for sota_comparison experiment. This is adjusted automatically based on the model.'
        assert args.window_size is None, 'Window size must be None.'
        assert args.seed is None, 'Seed must be None.'

        # Set configurations
        default_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'defaults.yml')
        arch_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'archs', f'{args.model}.yml')
        model_config = os.path.join(args.configs_dir, exp_folders[args.exp], 'model_opts', f'{args.model}.yml')
        configs = [default_config, arch_config, model_config]

        # Load options
        opt = load_yaml_options(configs)

        # Set additional options
        opt['name'] = f'{args.exp}_{args.clip}_{args.model}'
        opt['path']['results_root'] = os.path.join(args.results_dir, opt['path']['results_root'])
        opt['datasets']['test']['clip_name'] = args.clip

        ckpt_files = {
            'DGN_single_frame': 'DGN_single_frame.pth',
            'DGN_multi_frame': 'DGN_multi_frame.pth',
            'EDVR': 'EDVR.pth',
            'VRT': 'VRT.pth',
            'RVRT': 'RVRT.pth',
            'Shiftnet_baseRGB': 'Shiftnet_baseRGB.pth',
            'Shiftnet_RGBD': 'Shiftnet_RGBD.pth'
        }
        opt['path']['pretrain_network_g'] = os.path.join(args.ckpt_dir, exp_folders[args.exp], ckpt_files[args.model])
    
    opt = parse(opt, is_train=False)

    # distributed settings
    opt['dist'] = False
    opt['rank'], opt['world_size'] = get_dist_info()
    # random seed
    seed = opt.get('manual_seed')
    set_random_seed(seed + opt['rank'])

    return opt


def parse_options(is_train=True):
    """
    Parse command line arguments and option YAML files.

    Args:
        is_train (bool): Indicate whether in training mode or not. Default is True.

    Returns:
        dict: Parsed options including distributed settings, rank, world size, and random seed.

    Command Line Arguments:
        --opt (str): Path to option YAML file. This argument is required and can be specified multiple times.
        --launcher (str): Job launcher type. Choices are 'none', 'pytorch', or 'slurm'. Default is 'none'.
        --local_rank (int): Local rank for distributed training. Default is 0.

    The function performs the following steps:
        1. Parses command line arguments.
        2. Loads and merges option YAML files.
        3. Sets up distributed training settings based on the launcher type.
        4. Retrieves and sets the rank and world size for distributed training.
        5. Sets a random seed for reproducibility, using a manually specified seed if provided, or generating one if not.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', action='append', required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = load_yaml_options(args.opt)
    opt = parse(opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def load_yaml_options(opt_paths):
    """
    Load options from YAML files and merge them into a single dictionary.
    
    Args:
        opt_paths (str or list): Option file path(s). Can be a single path or a list of paths.
    
    Returns:
        dict: Merged options dictionary.

    This function reads the option file(s) specified in `opt_paths`, merges them if multiple files are provided,
    and returns a single dictionary containing all the options. The function uses OmegaConf to handle the YAML files
    and convert them into a dictionary format.
    """
    # Load options from YAML file
    if isinstance(opt_paths, str):
        opt_paths = [opt_paths]

    for i, opt_path in enumerate(opt_paths):
        with open(opt_path, mode='r') as f:
            if i == 0:
                opt = OmegaConf.load(f)
            else:
                opt = OmegaConf.merge(opt, OmegaConf.load(f))
    
    # Convert OmegaConf to dictionary
    opt = OmegaConf.to_container(opt, resolve=True)

    # Add opt_path to the options
    opt['opt_path'] = opt_paths

    return opt


def parse(opt, is_train=True):
    """
    Initialize configuration options.

    Args:
        opt (dict): Options dictionary containing configuration for the experiment.
        is_train (bool): Indicate whether in training mode or not. Default is True.

    Returns:
        dict: Parsed and updated options dictionary.

    This function initializes various configuration options for the experiment, including:
        - Model type
        - Input and output sequence lengths
        - Model stride
        - Datasets
        - Network data
        - Paths for saving results and logs
        - Random seed
        - Debug mode settings

    It also checks for specific conditions related to the input and output sequence lengths and model types.
    The function ensures that the options are set correctly for both training and testing phases.
    It also handles the paths for saving results, logs, and options.
    The function is designed to be called at the beginning of the training or testing process to set up the environment
    and configurations.
    """

    opt['is_train'] = is_train
    
    # model stride
    if 'model_stride' not in opt:
        opt['model_stride'] = (opt['in_seq_length'] - opt['out_seq_length']) // 2

    # datasets
    if 'datasets' in opt:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if 'in_seq_length' in opt:
                dataset['in_seq_length'] = opt['in_seq_length']
            if 'out_seq_length' in opt:
                dataset['out_seq_length'] = opt['out_seq_length']
            if 'aux_data_model' in opt:
                dataset['aux_data_model'] = opt['aux_data_model']
            if 'model_type' in opt:
                dataset['model_type'] = opt['model_type']
            if 'full_depth_mode' in opt:
                if dataset.get('cropping', None) is not None:
                    dataset['cropping']['full_depth_mode'] = opt['full_depth_mode']

            dataset['model_stride'] = opt['model_stride']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

            # Not use local scratch in debug mode
            if 'debug' in opt['name']:
                dataset['use_local_scratch'] = False

            if phase == 'val':
                # cropping in validation phase
                if dataset.get('cropping') is not None:
                    if dataset['cropping'].get('type') == 'grid':
                        opt['val']['crop_size'] = dataset['cropping']['gt_patch_size']
                        # Use one patch in debug mode
                        if 'debug' in opt['name']:
                            dataset['cropping']['num_patches_per_side'] = 1
                if 'debug' in opt['name']:
                    # opt['datasets']['val']['n_frames_per_video'] = opt['out_seq_length']
                    opt['datasets']['val']['n_frames_per_video'] = opt['in_seq_length']


    # network data
    if 'aux_data_model' in opt and opt['model_type'] in ['DavideModel']:
        opt['network_g']['aux_data'] = opt['aux_data_model'] if opt['aux_data_model'] is not None else []
    if 'out_seq_length' in opt and opt['model_type'] in ['DavideModel']:
        opt['network_g']['out_frames'] = opt['out_seq_length'] if opt['out_seq_length'] is not None else opt['in_seq_length'] - 2
    if 'full_depth_mode' in opt and opt['model_type'] in ['DavideModel']:
        opt['network_g']['full_depth_mode'] = opt['full_depth_mode'] if opt['full_depth_mode'] is not None else False
    if 'model_stride' in opt and opt['model_type'] in ['DavideModel']:
        opt['network_g']['model_stride'] = opt['model_stride'] if opt['model_stride'] is not None else 0

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        if opt['path'].get('experiments_root') is None:
            experiment_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
            opt['path']['experiments_root'] = osp.join(opt['path']['root'], 'experiments')
        else:
            experiment_root = osp.join(opt['path']['experiments_root'], opt['name'])
        opt['path']['models'] = osp.join(experiment_root, 'models')
        opt['path']['training_states'] = osp.join(experiment_root,
                                                  'training_states')
        opt['path']['log'] = experiment_root
        opt['path']['visualization'] = osp.join(experiment_root,
                                                'visualization')
        opt['path']['options'] = osp.join(experiment_root, 'options')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val'].pop('val_freq_in_epochs', None)
                opt['val']['val_freq'] = 4
            if 'train' in opt:
                opt['train'].pop('total_epochs', None)
                opt['train']['total_iter'] = 7
            opt['logger']['print_freq'] = 1
            opt['logger'].pop('save_checkpoint_freq_in_epochs', None)
            opt['logger']['save_checkpoint_freq'] = 6
    else:  # test
        if opt['path'].get('results_root') is None:
            results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        else:
            results_root = osp.join(opt['path']['results_root'], opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
        opt['path']['options'] = osp.join(results_root, 'options')
        opt['path']['csv_results'] = osp.join(results_root, 'results')

    return opt


def update_options(opt, total_epochs, total_iters):
    """
    Update options to convert epoch-based configurations to iteration-based configurations.

    Args:
        opt (dict): Options dictionary containing configuration for the experiment.
        total_epochs (int): Total number of epochs for the training.
        total_iters (int): Total number of iterations for the training.

    Returns:
        dict: Updated options dictionary with iteration-based configurations.

    This function updates the options dictionary by converting epoch-based configurations to iteration-based configurations. 
    It recursively updates the number of iterations based on the number of iterations per epoch and modifies keys ending with
    '_epoch' and '_in_epochs' to their iteration-based counterparts.
    """
    # Recursive function to update the number of iterations based on the number of iterations per epoch
    def update_iters(opt, num_iters_per_epoch):
        _opt = {}
        for key, value in opt.items():
            if isinstance(value, DictConfig) or isinstance(value, dict):
                _opt[key] = update_iters(value.copy(), num_iters_per_epoch)  # Copy the inner dictionary
            elif '_epoch' in key:
                _key = key.split('_epoch')[0] + '_iter'
                _opt[_key] = value * num_iters_per_epoch if value != -1 else -1
            else:
                _opt[key] = value
        return _opt

    # Funtion to update "*_in_epochs" to iterations
    def update_epochs_to_iters(opt, num_iters_per_epoch):
        keys_to_update = []
        for key, value in opt.items():
            if isinstance(value, DictConfig) or isinstance(value, dict):
                update_epochs_to_iters(value, num_iters_per_epoch)
            elif '_in_epochs' in key:
                keys_to_update.append(key)

        for key in keys_to_update:
            value = opt[key]
            _key = key.split('_in_epochs')[0]
            if isinstance(value, list):
                opt[_key] = [v*num_iters_per_epoch for v in value]
            else:
                opt[_key] = value * num_iters_per_epoch
            opt.pop(key)

    # update train options
    num_iters_per_epoch = total_iters // total_epochs
    if 'scheduler' in opt['train'].keys():
        lr_scheduler = opt['train']['scheduler']
        update_epochs_to_iters(lr_scheduler, num_iters_per_epoch)
    train_opt = update_iters(opt['train'], num_iters_per_epoch)
    opt['train'] = train_opt
    
    # update validation options
    if 'val' in opt.keys():
        val_opt = opt['val']
        update_epochs_to_iters(val_opt, num_iters_per_epoch)
    
    # update logger options
    if 'logger' in opt.keys():
        logger_opt = opt['logger']
        update_epochs_to_iters(logger_opt, num_iters_per_epoch)
    
    # save updated options
    save_config(opt)
    
    return opt
