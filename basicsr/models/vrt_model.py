# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is modified from:
# - KAIR (https://github.com/cszn/KAIR.git)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import torch
from collections import OrderedDict
from basicsr.utils import get_root_logger
from .video_model import VideoModel


class VRTModel(VideoModel):
    """Model for Video Restoration Transformer architectures.
    
    This class implements the necessary methods for training and testing VRT and RVRT architectures.
    """

    def __init__(self, opt):
        """Initialize the VRTModel class.
        
        Args:
            opt (dict): Configuration. It contains:
                network_g (dict): Configuration of the generator network.
                train (dict): Training settings.
                val (dict): Validation settings.
                test (dict): Testing settings.
                datasets (dict): Configuration of the datasets.
                path (dict): Path options.
                logger (dict): Logging options.
                dist (dict): Distributed training options.
        """
        super(VRTModel, self).__init__(opt)
        if self.is_train:
            train_opt = self.opt['train']
            self.fix_iter = train_opt.get('fix_iter', 0)
            self.fix_keys = train_opt.get('fix_keys', [])
            self.fix_unflagged = True

    def setup_optimizers(self):
        """Initialize optimizers for training.
        
        This method allows for multiple learning rates for key parameters in the list `fix_keys`.
        If `fix_lr_mul` is set to 1, the learning rate for the fixed keys is the same as the rest of the network.
        Otherwise, the learning rate for the fixed keys is multiplied by `fix_lr_mul`.
        Multiple learning rates are only applied if `fix_iter` is set to a value greater than 0.
        """
        train_opt = self.opt['train']
        self.fix_keys = train_opt.get('fix_keys', [])

        if train_opt.get('fix_iter', 0) and len(self.fix_keys) > 0:
            fix_lr_mul = train_opt['fix_lr_mul']
            self.logger.info(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            if fix_lr_mul == 1:
                optim_params = self.net_g.parameters()
            else:  # separate dcn params and normal params for different lr
                normal_params = []
                flow_params = []
                for name, param in self.net_g.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                    else:
                        normal_params.append(param)
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['optim_g']['lr']
                    },
                    {
                        'params': flow_params,
                        'lr': train_opt['optim_g']['lr'] * fix_lr_mul
                    },
                ]

            optim_type = train_opt['optim_g'].pop('type')
            self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
        else:
            super(VRTModel, self).setup_optimizers()

    def optimize_parameters(self, current_iter):
        """Optimize the model's parameters.
        
        Args:
            current_iter (int): Current iteration.
        
        This method allows to fix key parameters for a certain number of iterations.
        If `fix_iter` is set, the key parameters in the list `fix_keys` will be fixed for the first `fix_iter` iterations.
        After that, all the parameters will be trained.
        """

        if self.fix_iter:
            if self.fix_unflagged and current_iter < self.fix_iter+1:
                self.logger.info(f'Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.')
                self.fix_unflagged = False
                for name, param in self.net_g.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_iter == self.fix_iter+1:
                self.logger.info(f'Train all the parameters from {self.fix_iter} iters.')
                self.net_g.requires_grad_(True)

        super(VRTModel, self).optimize_parameters(current_iter)

    def get_inputs(self, float_format=False):
        """Get the inputs for the model.
        
        Args:
            float_format (bool, optional): Whether to convert the inputs to float format.
                Default: False.
        
        Returns:
            inputs (list): The list of inputs.
        """
        if float_format:
            return self.lq.float()
        else:
            return self.lq

    def load_network(self, network, load_path, strict=True, param_key='params'):
        """Load network parameters from a specified path.
        
        Args:
            network (nn.Module): Network to load the parameters.
            load_path (str): Path to the state_dict.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by `network.state_dict()`.
                Default: True.
            param_key (str, optional): Key in the state_dict to load. Default: '
        """
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        self._print_different_keys_loading(network, state_dict, strict)
        network.load_state_dict(state_dict, strict=strict)


