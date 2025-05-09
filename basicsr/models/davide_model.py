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
import torch
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

from basicsr.models.archs import define_network
from basicsr.models.video_model import VideoModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, PatchesToImage, nullcontext


# Import basicsr modules: losses and metrics
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class DavideModel(VideoModel):
    """Model for Depth-Aware Video Deblurring (DAVIDE) task.
    
    The model is based on `VideoModel`, but with some additional features. 
    It integrates depth information to enhance video deblurring, loads a test local converter model (TLC) for inference, 
    uses Automatic Mixed Precision (AMP) to optimize training performance, 
    and applies multiple learning rates to effectively fuse depth and RGB features.
    """

    def __init__(self, opt):
        """Initialize the DavideModel class
        
        Args:
            opt (dict): Configuration. It contains:
                network_g (dict): Generator network settings. For DAVIDE task, it must contain:
                    aux_data (list of str): List of auxiliary data to be used in the model. 
                                            For example: ['depth'].
                train (dict): Training settings.
                val (dict): Validation settings.
                test (dict): Testing settings.
                datasets (dict): Configuration of the datasets.
                path (dict): Path options.
                logger (dict): Logging options.
                dist (dict): Distributed training options.
        """
        # Set auxilary data
        self.aux_data = opt['network_g'].get('aux_data', [])
        super(DavideModel, self).__init__(opt)

    def load(self):
        """Load model's pretrained weights and optionally initialize the 
        Test Local Converter (TLC) if specified in the configuration.

        This method calls the parent class's load method. If TLC is to be used, 
        it logs the usage, modifies the network configuration, defines and initializes
        the TLC network, and sets it to evaluation mode. If not in training mode and TLC is used, 
        it removes the main network (net_g) to free up resources.
        """
        super(DavideModel, self).load()

        # Load test local converter model
        use_tlc = self.opt['val'].get('use_TLC', False) if self.is_train else self.opt['test'].get('use_TLC', False)
        if use_tlc:
            self.logger.info('Use Test Local Converter')
            tlc_layers = self.opt['val'].get('tlc_layers', ['local_avgpool']) if self.is_train else self.opt['test'].get('tlc_layers', ['local_avgpool'])
            self.logger.info(f'Test Local Converter layers: {tlc_layers}')
            network_opt = self.opt['network_g'].copy()
            network_opt['type'] = 'Local' + network_opt['type']
            self.net_g_TLC = define_network(network_opt).to(self.device)
            self.net_g_TLC.init_TLC(tlc_layers)
            self.model_TLC()
            self.net_g_TLC.eval()
        
        # Remove net_g if not in training mode and using TLC
        if not self.is_train and use_tlc:
            del self.net_g
    
    def model_TLC(self):
        """Copy the weights of the main network (net_g) to the Test Local Converter (TLC) network."""
        net_g =self.get_bare_model(self.net_g)
        net_g_params = dict(net_g.named_parameters())
        net_g_TLC_params = dict(self.net_g_TLC.named_parameters())

        for k in net_g_TLC_params.keys():
            net_g_TLC_params[k].data.copy_(net_g_params[k].data)

    def setup_optimizers(self):
        """Initialize optimizers for training.
        
        This method overrides the parent class's method to allow for multiple learning rates
        for the fusion of depth and rgb features. If the `fusion_lr_mul` is set to a value other than 1,
        the learning rate for the fusion layer is multiplied by that value.
        """
        train_opt = self.opt['train']
        fusion_lr_mul = train_opt.get('fusion_lr_mul', 1)
        if fusion_lr_mul == 1:
            return super().setup_optimizers()
        else:
            self.logger.info(f'Multiple the learning rate for fusion with {fusion_lr_mul}.')
            normal_params = []
            fusion_params = []
            for name, param in self.net_g.named_parameters():
                if '.fusion.' in name:
                    fusion_params.append(param)
                else:
                    normal_params.append(param)
            
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': fusion_params,
                    'lr': train_opt['optim_g']['lr'] * fusion_lr_mul
                },
            ]

            optim_type = train_opt['optim_g'].pop('type')
            self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
            self.optimizers.append(self.optimizer_g)
    
    def setup_losses(self):
        """Setup the losses for the model.
        
        This method overrides the parent class's method to allow for the use of depth information in the loss function.
        If the `depth_weight` is set in the configuration, the depth information is used to weight the loss function.
        The `depth_weight` is calculated based on the type and reduction method specified in the configuration.
        """
        super(DavideModel, self).setup_losses()
        if self.opt['train'].get('depth_weight', None) is not None:
            self.logger.info(f"Using depth weight in loss with type {self.opt['train']['depth_weight']['type']} (reduction: {self.opt['train']['depth_weight']['reduction']}), starting at iter {self.opt['train']['depth_weight']['start_iter']}.")

    def feed_data(self, data):
        """Feed the data to the model.
        
        This method overrides the parent class's method to allow for the use of depth information.
        If the 'depth', 'mono_depth_sharp', 'mono_depth_blur' or is specified in the configuration, 
        the depth information is fed to the model.
        
        Args:
            data (dict): The data to be fed to the model.    
        """
        sample, sample_info = data
        lq, gt, = sample['blur'], sample['gt']

        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)

        if 'depth' in self.aux_data:
            self.depth = sample['depth'].to(self.device)
        if 'mono_depth_sharp' in self.aux_data:
            self.depth = sample['mono_depth_sharp'].to(self.device)
        if 'mono_depth_blur' in self.aux_data:
            self.depth = sample['mono_depth_blur'].to(self.device)
        # if 'conf' in self.aux_data:
        #     self.conf = sample['conf'].to(self.device).half()
        
        # set data to half precision if AMP is enabled
        if self.enable_AMP:
            self.lq = self.lq.half()
            # self.gt = self.gt.half()
            if 'depth' in self.aux_data or 'mono_depth_sharp' in self.aux_data or 'mono_depth_blur' in self.aux_data:
                self.depth = self.depth.half()
            # if 'conf' in self.aux_data:
            #     self.conf = self.conf.half()
    
    def depth_weight(self, depth, current_iter):
        """Computes a depth-based weight for the loss function.
        
        Args:
            depth (torch.Tensor): The depth tensor.
            current_iter (int): The current iteration.
            
        Returns:
            torch.Tensor or None: The computed weight tensor or None if depth weight is not used.

        This function computes a loss weight tensor based on the depth information. The weight tensor is computed
        based on the type and reduction method specified in the configuration. Weight types include: 'var' and 'roughness'.
        """
        if self.opt['train'].get('depth_weight', None) is None:
            return None
        else:
                
            if current_iter < self.opt['train']['depth_weight']['start_iter']:
                return None
            else:
                stencil =  (depth.shape[1] - self.opt['out_seq_length']) // 2
                if stencil > 0:
                    depth = depth[:, stencil:-stencil]
                               
                if self.opt['train']['depth_weight']['type'] == 'var':
                    weight = self.opt['train']['depth_weight']['weight']
                    return 1.0 + weight * depth.var(dim=(3, 4), keepdim=True)
                
                elif self.opt['train']['depth_weight']['type'] == 'roughness':
                    weight = self.opt['train']['depth_weight']['weight']
                    laplacian_filter = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]).to(depth.device)

                    B, F, C, H, W = depth.shape
                    depth = depth.view(B*F, C, H, W)
                    depth_lap = torch.nn.functional.conv2d(depth, laplacian_filter, padding='same')
                    depth_lap = depth_lap.view(B, F, C, H, W)

                    if self.opt['train']['depth_weight']['reduction'] == 'mean':
                        return 1.0 + weight * (depth_lap ** 2).mean(dim=(3, 4), keepdim=True) 
                    
                    elif self.opt['train']['depth_weight']['reduction'] == 'none':
                        return 1.0 + weight * (depth_lap ** 2)
                    
                    else:
                        raise NotImplementedError(f"Reduction method {self.opt['train']['depth_weight']['reduction']} not implemented.")

    def get_inputs(self, float_format=False):
        """Get the inputs for the model.
        
        Args:
            float_format (bool, optional): Whether to return the inputs in float format. Default is False.
        
        Returns:
            inputs (list): The list of inputs.

        This method overrides the parent class's method to allow for the use of depth information.
        If the 'depth', 'mono_depth_sharp', 'mono_depth_blur' or is specified in the configuration,
        the depth information is included in the inputs.
        """
        inputs = [self.lq]
        if 'depth' in self.aux_data or 'mono_depth_sharp' in self.aux_data or 'mono_depth_blur' in self.aux_data:
            inputs.append(self.depth)
            
        if float_format:
            inputs = [i.float() if i is not None else None for i in inputs]
        return inputs
    
    def loss_calculation(self, current_iter):
        """Calculate the loss function.

        Args:
            current_iter (int): The current iteration.

        Returns:
            l_total (torch.Tensor): The total loss.
            loss_dict (OrderedDict): The dictionary containing the individual losses.
        
        This method overrides the parent class's method to allow for the use of depth information in the loss function.
        """
        l_total = 0
        loss_dict = OrderedDict()

        # weight
        if self.opt['train'].get('depth_weight', None) is not None:
            weight = self.depth_weight(self.depth, current_iter)
        else:
            weight = None

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt, weight)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            if len(self.output.shape) == 5:
                B, F, C, H, W = self.output.shape
                self.output = self.output.reshape(B*F, C, H, W)
                self.gt = self.gt.reshape(B*F, C, H, W)
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        return l_total, loss_dict

    def optimize_parameters(self, current_iter):
        """Optimize the model's parameters.
        
        Args:
            current_iter (int): The current iteration.
        """
        super(DavideModel, self).optimize_parameters(current_iter)

        # Update TLC if used and it is time to do so
        if self.opt['val'].get('use_TLC', False):
            if current_iter % self.opt['val']['val_freq'] == 0:
                self.model_TLC()

    def test(self, inputs=None):
        """Single test iteration.
        
        Args:
            inputs (list, optional): The inputs to be tested. Default is None.

        This method overrides the parent class's method to allow for the use of the Test Local Converter (TLC) network and Exponential Moving Average (EMA) network.
        """
        if inputs is None:
            inputs = self.get_inputs()

        if hasattr(self, 'net_g_ema'):
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.enable_AMP else nullcontext():
                    self.output = self.net_g_ema(inputs)
        elif hasattr(self, 'net_g_TLC'):
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.enable_AMP else nullcontext():
                    self.output = self.net_g_TLC(inputs)
        else:
            self.net_g.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.enable_AMP else nullcontext():
                    self.output = self.net_g(inputs)
            self.net_g.train()

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network parameters from a specified path.

        Args:
            net (nn.Module): The network to load the parameters into.
            load_path (str): The path of the network parameters to be loaded.
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the keys 
                        in `net`'s state_dict. Default is True.
            param_key (str, optional): The key under which the parameters are stored in the loaded file. 
                            If set to None, use the root of the loaded file. Default is 'params'.
        """
        net = self.get_bare_model(net)
        self.logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
