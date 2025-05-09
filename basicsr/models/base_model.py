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
import logging
import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.dist_util import master_only
from basicsr.utils import get_root_logger


class BaseModel():
    """Base model for image and video restoration.
    
    This class serves as a base for all restoration models. It provides common 
    functionalities and interfaces for training, testing, saving, loading, etc.
    """
    def __init__(self, opt):
        """Initialize the BaseModel class.

        Args:
            opt (dict): Configuration dictionary for the model. It should contain:
                - 'is_train' (bool): Whether the model is in training mode.
                - 'num_gpu' (int): Number of GPUs.
                - 'dist' (bool): Whether to use distributed training.
                - 'train' (dict, optional): Training options.
                - 'val' (dict, optional): Validation options.
                - 'test' (dict, optional): Test options.

        """
        self.opt = opt
        self.logger = get_root_logger()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        """Feed data to the model.

        Args:
            data (dict): Data dictionary.
        """
        pass

    def optimize_parameters(self):
        """Optimize parameters."""
        pass

    def get_current_visuals(self):
        """Get current visual results."""
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=True, use_image=True, epoch=0):
        """Validation function.

        This function performs validation using the provided dataloader. It supports both distributed and non-distributed
        validation based on the configuration.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration number.
            tb_logger (tensorboard.SummaryWriter): Tensorboard logger for logging validation metrics.
            save_img (bool, optional): Whether to save validation images. Default is False.
            rgb2bgr (bool, optional): Whether to convert images from RGB to BGR format before saving. Default is True.
            use_image (bool, optional): Whether to use saved images to compute metrics (e.g., PSNR, SSIM). If False, metrics
                                        are computed directly from the network's output. Default is True.
            epoch (int, optional): Current epoch number. Default is 0.

        Returns:
            dict: Validation results.
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, epoch)
        else:
            return self.nondist_validation(dataloader, current_iter, tb_logger,
                                    save_img, rgb2bgr, use_image, epoch)
    
    def model_ema(self, decay=0.999):
        """Exponential moving average for model weights.

        Args:
            decay (float, optional): Decay rate. Default is 0.999.

        This function updates the EMA of the model parameters using the specified decay rate. The EMA helps in stabilizing
        the training process and can be used for evaluation to get smoother results.
        """
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def get_current_log(self):
        """Get current log."""
        return self.log_dict

    def model_to_device(self, net):
        """Move the model to the specified device and wrap it with DistributedDataParallel or DataParallel if necessary.

        This function moves the given model to the device specified in the configuration. If distributed training is enabled,
        it wraps the model with `DistributedDataParallel`. If multiple GPUs are available but distributed training is not
        enabled, it wraps the model with `DataParallel`.

        Args:
            net (nn.Module): The neural network model to be moved to the device.

        Returns:
            nn.Module: The model moved to the specified device and wrapped with the appropriate parallelism strategy if necessary.

        Notes:
            - If `dist` is set to True in the configuration, the model will be wrapped with `DistributedDataParallel`.
            - If `num_gpu` is greater than 1 and `dist` is set to False, the model will be wrapped with `DataParallel`.
            - The `find_unused_parameters` option can be set in the configuration to handle models with unused parameters.
            - The `use_static_graph` option can be set in the configuration to use a static graph for `DistributedDataParallel`.
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', True)
            use_static_graph = self.opt.get('use_static_graph', False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                self.logger.warning(f'Using static graph. Make sure that "unused parameters" will not change during training loop.')
                net._set_static_graph()
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net
    
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        """Get the optimizer based on the specified type.

        This function returns an optimizer instance based on the provided type and parameters. It supports various types of
        optimizers available in PyTorch.

        Args:
            optim_type (str): Type of the optimizer. Supported types are 'Adam', 'AdamW', 'Adamax', 'SGD', 'ASGD', 'RMSprop', and 'Rprop'.
            params (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
            lr (float): Learning rate for the optimizer.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            torch.optim.Optimizer: An instance of the specified optimizer.

        Raises:
            NotImplementedError: If the specified optimizer type is not supported.

        Example:
            >>> optimizer = self.get_optimizer('Adam', model.parameters(), lr=0.001, betas=(0.9, 0.999))
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """Set up learning rate schedulers for the optimizers.

        This function initializes and sets up the learning rate schedulers for the optimizers based on the configuration
        specified in `self.opt['train']['scheduler']`. It supports various types of schedulers, including:

        - MultiStepLR
        - MultiStepRestartLR
        - CosineAnnealingRestartLR
        - CosineAnnealingLR
        - LinearLR
        - LinearDecreaseWithPlateauLR
        - VibrateLR

        The appropriate scheduler is selected based on the 'type' field in the scheduler configuration. The schedulers are
        then appended to `self.schedulers`.

        Raises:
            NotImplementedError: If the specified scheduler type is not supported.

        Example:
            >>> self.setup_schedulers()
        """
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'LinearDecreaseWithPlateauLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearDecreaseWithPlateauLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get the bare model from a wrapped model.

        This function retrieves the underlying model from a model that is wrapped with `DistributedDataParallel` or 
        `DataParallel`. This is useful when you need to access the original model's attributes or methods that are not 
        directly accessible through the wrapper.

        Args:
            net (nn.Module): The neural network model, which may be wrapped with `DistributedDataParallel` or `DataParallel`.

        Returns:
            nn.Module: The bare (unwrapped) model.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the class name and the total number of parameters of the given network. 

        Args:
            net (nn.Module): The neural network model to be printed. It can be wrapped with `DataParallel` or 
                            `DistributedDataParallel`.
    """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        self.logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        self.logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rates for each parameter group in the optimizers.

        This function sets the learning rates for each parameter group in the optimizers, typically used during the warmup phase.

        Args:
            lr_groups_l (list): List of learning rate groups, where each sublist corresponds to an optimizer's parameter groups.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial learning rates set by the schedulers.

        This function retrieves the initial learning rates for each parameter group in all optimizers. The initial learning
        rates are those set by the learning rate schedulers at the beginning of training.

        Returns:
            list: A list of lists, where each sublist contains the initial learning rates for the parameter groups of an optimizer.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update the learning rate based on the current iteration.

        This function updates the learning rate using the schedulers. If warmup iterations are specified, it performs a 
        linear warmup of the learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Number of warmup iterations. Use -1 for no warmup. Default is -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        """Get the current learning rate of the first optimizer.

        Returns:
            list: A list of current learning rates for each parameter group in the first optimizer.
        """
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save network(s) to disk.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iteration number. Use -1 for 'latest' and -2 for 'best'.
            param_key (str | list[str]): The parameter key(s) to save the network. Default is 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        if current_iter == -2:
            current_iter = 'best'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                self.logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            self.logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            self.logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                self.logger.warning(f'  {v}')
            self.logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                self.logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    self.logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load a network from a specified path.

        Args:
            net (nn.Module): The network to load the parameters into.
            load_path (str): The path of the network to be loaded.
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the keys returned by `net`'s `state_dict` function. Default is True.
            param_key (str, optional): The parameter key of the loaded network. If set to None, use the root 'path'. Default is 'params'.

        """
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                self.logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        self.logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training for resuming later.

        This function saves the current training state, including the epoch, iteration, optimizer states, scheduler states,
        and other relevant metrics. This is useful for resuming training from the saved state.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration. Use -1 for 'latest' and -2 for 'best'.
        """
        if current_iter != -1 or current_iter != -2:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            
            # save best metric info
            if hasattr(self, 
                       'best_metric_name') and hasattr(self, 
                                                       'best_metric_value') and hasattr(self, 
                                                                                        'best_iter'):    
                state['best_metric_name'] = self.best_metric_name
                state['best_metric_value'] = self.best_metric_value
                state['best_iter'] = self.best_iter
            
            # save flops and total params
            if hasattr(self, 'flops'):
                state['flops'] = self.flops
            if hasattr(self, 'total_params'):
                state['total_params'] = self.total_params
            
            # save wandb id
            if self.opt['logger'].get('wandb') is not None:
                state['wandb_id'] = self.opt['logger']['wandb']['resume_id']

            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'],
                                     save_filename)
            torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Reload the optimizers, schedulers, and other relevant metrics for resumed training.

        This function reloads the state of the optimizers and schedulers from a given resume state. It also restores
        additional metrics such as the best metric name, best metric value, best iteration, FLOPs, and total parameters
        if they are present in the resume state.

        Args:
            resume_state (dict): A dictionary containing the state to resume from, including optimizers, schedulers, and
                                other relevant metrics.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        if hasattr(self, 
                'best_metric_name') and hasattr(self, 
                                                'best_metric_value') and hasattr(self, 
                                                                                'best_iter'):
            if 'best_metric_name' in resume_state:
                self.best_metric_name = resume_state['best_metric_name']
            if 'best_metric_value' in resume_state:
                self.best_metric_value = resume_state['best_metric_value']
            if 'best_iter' in resume_state:
                self.best_iter = resume_state['best_iter']

        if hasattr(self, 'flops'):
            self.flops = resume_state['flops']

        if hasattr(self, 'total_params'):
            self.total_params = resume_state['total_params']              


    def reduce_loss_dict(self, loss_dict):
        """Reduce and average the loss dictionary across multiple GPUs.

        In distributed training, this function averages the losses among different GPUs.

        Args:
            loss_dict (OrderedDict): Dictionary containing the loss values.

        Returns:
            OrderedDict: Dictionary with the averaged loss values.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                if self.device.type == 'cuda':  # Check if the current device is a CUDA device
                    torch.cuda.synchronize()  # Ensure all CUDA operations are completed
                torch.distributed.reduce(losses, dst=0)
                losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
