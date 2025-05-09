# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is modified from:
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
from basicsr.utils import get_root_logger
from .video_model import VideoModel


class EDVRModel(VideoModel):
    """
    Model for EDVR network.

    This model implements the EDVR (Enhanced Deformable Video Restoration) network, 
    which is designed for video restoration tasks using enhanced deformable convolutional 
    networks.

    Reference:
        Wang, Xintao, et al. "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks." 
        Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 2019.
    """

    def __init__(self, opt):
        """Initialize the EDVRModel class.
        
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
        super(EDVRModel, self).__init__(opt)
        if self.is_train:
            self.train_tsa_iter = opt['train'].get('tsa_iter')

    def setup_optimizers(self):
        """Initialize optimizers for training.
        
        This method allows for multiple learning rates for the deformable convolutional.
        If `dcn_lr_mul` is set to 1, the learning rate for the deformable convolutional.
        Otherwise, the learning rate for the deformable convolutional is multiplied by `dcn_lr_mul`.
        
        """
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        self.logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        if dcn_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        """Optimize the model's parameters.

        Args:
            current_iter (int): Current iteration.
        
        This method allows for training the TSA module for a certain number of iterations.
        If `train_tsa_iter` is set, the TSA module will be trained for the first `train_tsa_iter` iterations.
        After that, all the parameters will be trained.
        """
        if self.train_tsa_iter:
            if current_iter == 1:
                logger = get_root_logger()
                logger.info(f'Only train TSA module for {self.train_tsa_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_tsa_iter:
                logger = get_root_logger()
                logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True

        super(EDVRModel, self).optimize_parameters(current_iter)

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
