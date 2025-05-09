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
import os

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import imwrite, tensor2img, nullcontext


# Import basicsr modules: losses and metrics
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class VideoModel(BaseModel):
    """Model for video restoration tasks.
    
    The class is built on top of `BaseModel`. This class defines the network, 
    loads pretrained models, sets up losses, optimizers, and schedulers, feeds data, optimizes 
    parameters, evaluates the model, retrieves current visuals, saves the model, and calculates 
    model size and FLOPs. It also includes methods for validation, testing on videos, and testing 
    on clips.
    """

    def __init__(self, opt):
        """Initialize the VideoModel class.
        
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
        super(VideoModel, self).__init__(opt)

        # # Copy aux_data 
        # self.aux_data = opt['network_g'].get('aux_data', [])

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # disable Automatic Mixed Precision (AMP) by default (enable it in training settings)
        self.enable_AMP = False  
       
        # load pretrained models
        self.load()
  

    def init_training_settings(self):
        """Initialize training settings."""

        train_opt = self.opt['train']
        
        # Set training mode
        self.net_g.train()

        # Set up loss, optimizers and schedulers
        self.setup_losses()
        self.setup_optimizers()
        self.setup_schedulers()

        # Init validation metrics
        self.init_validation_metrics()

        # Enable Automatic Mixed Precision (AMP)
        if train_opt.get('enable_AMP', False):
            self.enable_AMP = True
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info('Enable Automatic Mixed Precision (AMP)')
    
    def init_testing_settings(self):
        """Initialize testing settings."""

        test_opt = self.opt['test']
        
        if test_opt.get('enable_AMP', False):
            self.enable_AMP = True
            self.logger.info('Enable Automatic Mixed Precision (AMP)')


    def load(self):
        """Load model's pretrained weights and EMA weights.
        
        The function loads the pretrained weights for the generator network (net_g) 
        and the Exponential Moving Average (EMA) weights (net_g_ema).
        
        """
        phase_opt = self.opt['train'] if self.is_train else self.opt['test']

        # Load pretrained model for net_g
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # Load pretrained model for net_g_ema (if EMA is used)
        self.ema_decay = phase_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            self.logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(deepcopy(self.opt['network_g'])).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

    def setup_losses(self):
        """Initialize losses for training.
        
        The function initializes pixel and perceptual losses for training.
        Pixel loss is used to measure the difference between the output and the ground truth, stored in `self.cri_pix`.
        Perceptual loss is used to measure the difference between the output and the ground truth in a feature space, stored in `self.cri_perceptual`.
        """
        train_opt = self.opt['train']
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def init_validation_metrics(self):
        """Initialize validation metrics."""
        if self.opt.get('val', None) is not None:
            if self.opt['val'].get('val_freq', np.inf) < self.opt['train'].get('total_iter'):
                self.best_iter = 0
                self.best_metric_value = None
                self.flops = None
                self.total_params = None
                self.best_metric_name = list(self.opt['val']['metrics'].keys())[0]


    def setup_optimizers(self):
        """Initialize optimizers for training."""
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """Feed data to the model.
        
        Args:
            data (dict): Data dictionary.
            
        The function feeds the blurry/low-quality (lq) and ground-truth (gt) images to the model.
        """
        sample, sample_info = data
        lq, gt, = sample['blur'], sample['gt']

        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)

        # Set data to half precision if AMP is enabled
        if self.enable_AMP:
            self.lq = self.lq.half()
            
    def get_inputs(self, float_format=False):
        """Get the inputs for the model.
        
        Args:
            float_format (bool, optional): Whether to return the inputs in float format. Default is False.

        Returns:
            inputs (list): The list of inputs.
        """
        if float_format:
            return self.lq.float()
        else:
            return self.lq
    
    def loss_calculation(self, current_iter):
        """Calculate the loss function.

        Args:
            current_iter (int): The current iteration.

        Returns:
            l_total (torch.Tensor): The total loss.
            loss_dict (OrderedDict): The dictionary containing the individual losses.
        """
        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        return l_total, loss_dict
    
    def print_grad_norm(self, current_iter):
        """Print the gradients norm."""
        if current_iter % self.opt['logger']['print_freq'] == 0:
            grad_norm = 0.0
            for param in self.net_g.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item()**2
            grad_norm = grad_norm**0.5

            # Print gradients norm
            self.logger.info("Gradients Norm: ", grad_norm[0])

    def optimize_parameters(self, current_iter):
        """Optimize the model parameters.
        
        Args:
            current_iter (int): The current iteration.
            
        This function performs a forward pass to get the model output, calculates the loss, 
        performs a backward pass, applies Automatic Mixed Precision (AMP) scaling if enabled, 
        clips gradients if specified, updates the model weights using the optimizer, and updates 
        the Exponential Moving Average (EMA) weights if applicable.
        """
        self.optimizer_g.zero_grad()

        with torch.cuda.amp.autocast() if self.enable_AMP else nullcontext():
            # Forward
            inputs = self.get_inputs()
            self.output  = self.net_g(inputs)

            # Calculate loss
            l_total, loss_dict = self.loss_calculation(current_iter)

        # Backward and optimize
        if self.enable_AMP:
            self.scaler.scale(l_total).backward()
            self.scaler.unscale_(self.optimizer_g)
        else:
            l_total.backward()

        # self.print_grad_norm(current_iter)

        # Grad norm clipping
        if self.opt['train'].get('grad_clip', None) is not None:
            if isinstance(self.opt['train']['grad_clip'], bool):
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']['grad_clip'])

        # Update weights with optimizer
        if self.enable_AMP:
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self, inputs=None):
        """Single test iteration.
        
        Args:
            inputs (list, optional): The list of inputs. Default is None.

        The function performs a forward pass to get the model output.
        If inputs is None, the function uses the current `lq` tensor.
        If the model has an Exponential Moving Average (EMA) network, the function uses the EMA network to get the output.

        """
        if inputs is None:
            inputs = self.get_inputs()

        if hasattr(self, 'net_g_ema'):
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.enable_AMP else nullcontext():
                    self.output = self.net_g_ema(inputs)
        else:
            self.net_g.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.enable_AMP else nullcontext():
                    self.output = self.net_g(inputs)
            self.net_g.train()


    def get_latest_images(self):
        """Get the latest images.
        
        Returns:
            list: The list of latest images.
            
        The function returns the latest images in the following order: lq, gt, output."""
        return [self.lq[0].float(), self.gt[0].float(), self.output[0].float()]


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, epoch):
        """Validation with distributed settings.
        
        Args:
            dataloader (DataLoader): PyTorch dataloader.
            current_iter (int): The current iteration.
            tb_logger (SummaryWriter): Tensorboard logger.
            save_img (bool): Whether to save images.
            rgb2bgr (bool): Whether to convert RGB to BGR.
            use_image (bool): Whether to use image.
            epoch (int): The current epoch.
            
        Returns:
            float: The current metric value.
        """
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, epoch)
        else:
            return 0.

    def nondist_validation(self, test_loader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image, epoch):
        """Validation with non-distributed settings.
        
        Args:
            test_loader (DataLoader): PyTorch dataloader.
            current_iter (int): The current iteration.
            tb_logger (SummaryWriter): Tensorboard logger.
            save_img (bool): Whether to save images.
            rgb2bgr (bool): Whether to convert RGB to BGR.
            use_image (bool): Whether to use image.
            epoch (int): The current epoch.
        
        Returns:
            float: The current metric value.
        """
        import math
        dataset_name = test_loader.dataset.opt['name']
        save_img_freq = self.opt['val'].get('save_img_freq', -1)
        
        # Initialize metrics
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            # overall metrics
            self.metric_results = {
                metric: 0.0
                for metric in self.opt['val']['metrics'].keys()
            }
            # best metric
            self.best_metric_name = list(self.opt['val']['metrics'].keys())[0]
            # best metric value
            test_results = OrderedDict()
            mean_test_results = OrderedDict()
            for name in self.opt['val']['metrics'].keys():
                test_results[name] = []
                mean_test_results[name] = 0.0
        
        # Iterate over folders
        for idx, test_data in enumerate(test_loader):
            # Compute computational metrics
            if hasattr(self, 'total_params') and hasattr(self, 'flops'):
                if self.total_params is None and self.flops is None:
                    self.total_params = 0 if 'debug' in self.opt['name'] else self.get_model_size()
                    self.flops = 0 if 'debug' in self.opt['name'] else self.get_model_flops()

            # Inference
            self.feed_data(test_data)
            self.evaluate_model()

            # Get current visuals
            visuals = self.get_current_visuals(phase='val')
            input = visuals['lq']
            output = visuals['result']
            gt = visuals['gt'] if 'gt' in visuals else None
            folder = test_data[1]['folder']
            save_count = 0
            save_img = False

            # Clear memory
            self.clean_tensors()

            if with_metrics:
                test_results_folder = OrderedDict()
                for name in self.opt['val']['metrics'].keys():
                    test_results_folder[name] = []

            # Iterate over frames
            for i in range(output.shape[0]):
                if self.opt['val']['save_img'] and save_img_freq > 0 and i % (save_img_freq*self.opt['out_seq_length']) == 0 and save_count == 0:
                    save_img = True
                    save_count = self.opt['out_seq_length']

                if save_img and save_count == 0:
                    save_img = False
                elif save_img:
                    save_count -= 1

                # Prepare images
                sr_img = tensor2img(output[i], rgb2bgr=rgb2bgr, min_max=(0, 1), make_grid_on=False)
                lr_img = tensor2img(input[i], rgb2bgr=rgb2bgr, min_max=(0, 1), make_grid_on=False)
                if gt is not None:
                    gt_img = tensor2img(gt[i], rgb2bgr=rgb2bgr, min_max=(0, 1), make_grid_on=False)

                # Save images    
                if save_img:
                    if self.opt['datasets']['val'].get('concat_frames_in_batch', False):
                        seq_ = os.path.basename(test_data[1]['blur_path'][0][i]).split('.')[0]
                    else:
                        seq_ = os.path.basename(test_data[1]['blur_path'][i][0]).split('.')[0]
                    save_sr_path = osp.join(self.opt['path']['visualization'],
                                                f'{folder[0]}_{seq_}',
                                                f'out_iter{current_iter:06d}.png')
                    os.makedirs(osp.dirname(save_sr_path), exist_ok=True)
                    imwrite(sr_img, save_sr_path)
                    if current_iter==self.opt['val']['val_freq']:
                        save_lr_path = osp.join(self.opt['path']['visualization'],
                                                f'{folder[0]}_{seq_}',
                                                f'input.png')
                        save_gt_path = osp.join(self.opt['path']['visualization'],
                                                f'{folder[0]}_{seq_}',
                                                f'gt.png')
                        os.makedirs(osp.dirname(save_lr_path), exist_ok=True)
                        imwrite(lr_img, save_lr_path)
                        if gt is not None:
                            os.makedirs(osp.dirname(save_gt_path), exist_ok=True)
                            imwrite(gt_img, save_gt_path)

                # Calculate metrics per frame
                if with_metrics:
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        metric_mode = opt_.pop('mode')
                        test_results_folder[name].append(
                            getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                            )
      
            # Calculate metrics per clip
            log_msg = 'Testing {:20s}  ({:2d}/{}) '.format(folder[0], idx, len(test_loader))
            if with_metrics:
                if gt is not None:
                    for name in self.opt['val']['metrics'].keys():
                        metric_mean_per_clip = sum(test_results_folder[name]) / len(test_results_folder[name])
                        test_results[name].append(metric_mean_per_clip)
                        log_msg += f'{name.upper()}: {metric_mean_per_clip:.4f}; '
            self.logger.info(log_msg)

        # Summarize metrics
        if gt is not None:
            log_msg = 'Summary --> epoch:{:3d}, iter:{:8,d} Average '.format(epoch, current_iter)
            for name in self.opt['val']['metrics'].keys():
                mean_test_results[name] = sum(test_results[name]) / len(test_results[name])
                log_msg += f'{name.upper()}: {mean_test_results[name]:.4f}; '
            self.logger.info(log_msg)


        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] = mean_test_results[metric]
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
            
            # update best metric
            self._update_best_metric(current_iter, tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        """Log the validation metric values.
        
        Args:
            current_iter (int): The current iteration.
            dataset_name (str): The dataset name.
            tb_logger (SummaryWriter): Tensorboard logger.    
        """
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        self.logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def _update_best_metric(self, current_iter, tb_logger):
        """Update the best metric.
        
        Args:
            current_iter (int): The current iteration.
            tb_logger (SummaryWriter): Tensorboard logger.
        """
        # get comparison operator based om metric mode
        comp_opt = {
            'min': np.less,
            'max': np.greater
        }[self.opt['val']['metrics'][self.best_metric_name]['mode']]
        metric_value = self.metric_results[self.best_metric_name]
        if self.best_metric_value is None or comp_opt(metric_value,
                                                        self.best_metric_value):
            self.best_metric_value = metric_value
            self.best_iter = current_iter

            log_str = f'Best model:\n'
            log_str += f'\t # {self.best_metric_name}: {self.best_metric_value:.4f} (in iter {self.best_iter}) \n'
            log_str += f'\t # flops: {self.flops:.3e}, params: {self.total_params:.3e} \n'
            self.logger.info(log_str)

            # save best model
            self.save(epoch=-2, current_iter=-2) # -2 stands for the best model


    def get_current_visuals(self, phase='val'):
        """Get the current visuals.
        
        Args:
            phase (str, optional): The model mode. Default is 'val'."""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.float().detach().cpu()
        out_dict['result'] = self.output.float().detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.float().detach().cpu()
        
        if self.opt['datasets'][phase].get('concat_frames_in_batch', False) == False:
            for key, value in out_dict.items():
                out_dict[key] = value.squeeze(0)
        return out_dict
    
    def clean_tensors(self):
        """Clean buffer tensors.
        
        The function deletes the buffer tensors to free up memory.
        """
        del self.lq
        del self.output
        if hasattr(self, 'gt'):
            del self.gt
        torch.cuda.empty_cache()

    def save(self, epoch, current_iter):
        """Save model weights and training states.
        
        Args:
            epoch (int): The current epoch.
            current_iter (int): The current iteration.
        """
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def get_model_size(self):
        """Get the total number of parameters in the model."""
        # get total number of parameters
        total_num = sum(p.numel() for p in self.net_g.parameters())

        return total_num
    
    def get_model_flops(self):
        """Get the total number of FLOPs in the model."""
        from fvcore.nn import FlopCountAnalysis

        inputs = self.get_inputs(float_format=True)
        n = self.opt['in_seq_length']
        if isinstance(inputs, list):
            inputs = [i[...,0:n,:,0:256,0:256] if i is not None else None for i in inputs]
        else:
            inputs = inputs[...,0:n,:,0:256,0:256]
        
        self.net_g.eval()
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            flops = FlopCountAnalysis(self.net_g, inputs)
            total_flops = flops.total()
        
        self.net_g.train()

        return total_flops

    def evaluate_model(self, phase='val'):
        """Evaluate restoration model.
        
        Args:
            phase (str, optional): The model mode. Default is 'val'.
        """

        inputs = self.get_inputs()
        self.net_g.eval()

        pad_seq = self.opt[phase].get('pad_seq', False)
        model_stride = self.opt.get('model_stride', 0)

        if self.opt['datasets'][phase].get('concat_frames_in_batch', False):
            if isinstance(inputs, list):
                inputs = [i.unsqueeze(0) for i in inputs]
            else:
                inputs = inputs.unsqueeze(0)

        if pad_seq:
            pad_value = pad_seq if not isinstance(pad_seq, bool) else model_stride
            # reflect padding in dim 1
            if isinstance(inputs, list):
                inputs = [torch.cat([i[:,1:pad_value+1,:,:,:].flip(1), i, i[:,-pad_value-1:-1:,:,:,:].flip(1)], dim=1) for i in inputs]
            else:
                inputs = torch.cat([inputs[:,1:pad_value+1,:,:,:].flip(1), inputs, inputs[:,-pad_value-1:-1:,:,:,:].flip(1)], dim=1)

        # inference per video
        with torch.no_grad():
            self.output = self._test_video(inputs, phase)

        # if pad_seq:
        #     self.output = self.output[:, pad_value:-pad_value, :, :, :]
        
        if self.opt['datasets'][phase].get('concat_frames_in_batch', False):
            self.output = self.output.squeeze(0)


        self.net_g.train()

    def _test_video(self, inputs, phase='val'):
        """Test the video as a whole or as clips (divided temporally).
        
        Args:
            inputs (list): The list of inputs.
            phase (str, optional): The model mode. Default is 'val'.
        
        Returns:
            torch.Tensor: The output tensor.
        """

        num_frame_testing = self.opt[phase].get('num_frame_testing', 0)
        sf = self.opt['scale']
        model_stride = self.opt.get('model_stride', 0)

        if num_frame_testing:
            # test as multiple clips if out-of-memory
            num_frame_overlapping = self.opt[phase].get('num_frame_overlapping', 2)
            not_overlap_clip_border = self.opt[phase].get('not_overlap_clip_border', False)
            if isinstance(inputs, list):
                b, d_in, c, h, w = inputs[0].size()
            else:
                b, d_in, c, h, w = inputs.size()
            d_out = d_in - 2*model_stride
            stride_in = num_frame_testing - num_frame_overlapping
            stride_out = num_frame_testing - num_frame_overlapping
            d_in_idx_list = list(range(0, d_in-num_frame_testing-2*model_stride, stride_in)) + [max(0, d_in-num_frame_testing-2*model_stride)]
            d_out_idx_list = list(range(0, d_out-num_frame_testing, stride_out)) + [max(0, d_out-num_frame_testing)]
            E = torch.zeros(b, d_out, c, h*sf, w*sf)
            W = torch.zeros(b, d_out, 1, 1, 1)

            for d_in_idx, d_out_idx in zip(d_in_idx_list, d_out_idx_list):
                if isinstance(inputs, list):
                    inputs_clip = [i[:, d_in_idx:d_in_idx+num_frame_testing+2*model_stride, ...] for i in inputs]
                else:
                    inputs_clip = inputs[:, d_in_idx:d_in_idx+num_frame_testing+2*model_stride, ...]
                out_clip = self._test_clip(inputs_clip, phase)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d_out), 1, 1, 1))

                if not_overlap_clip_border:
                    if d_out_idx < d_out_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_out_idx > d_out_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_out_idx:d_out_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_out_idx:d_out_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = self.opt[phase].get('window_size', [6,8,8])
            if isinstance(inputs, list):
                d_in = inputs[0].size(1)
            else:
                d_in = inputs.size(1)
            d_pad = (d_in// window_size[0]+1)*window_size[0] - d_in
            if isinstance(inputs, list):
                inputs = [torch.cat([i, i[:,-d_pad:,:,:,:].flip(1)], 1) for i in inputs]
            else:
                inputs = torch.cat([inputs, inputs[:,-d_pad:,:,:,:].flip(1)], 1)
            
            output = self._test_clip(inputs, phase)
            d_out = d_in - 2*model_stride
            output = output[:, :d_out, :, :, :]

        return output
    
    def _test_clip(self, inputs, phase='val'):
        """Test the clip as a whole or as patches.
        
        Args:
            inputs (list): The list of inputs.
            phase (str, optional): The model mode. Default is 'val'.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        sf = self.opt['scale']
        window_size = self.opt[phase].get('window_size', [6,8,8])
        model_stride = self.opt.get('model_stride', 0)
        overlap_patch_size = self.opt[phase].get('overlap_size', 20)
        not_overlap_patch_border = self.opt[phase].get('not_overlap_border', True)
        size_patch_testing = self.opt[phase].get('size_patch_testing', 0)
        assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)

            # test patch by patch
            if isinstance(inputs, list):
                b, d_in, c, h, w = inputs[0].size()
            else:
                b, d_in, c, h, w = inputs.size()
            d_out = d_in - 2*model_stride
            stride = size_patch_testing - overlap_patch_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            E = torch.zeros(b, d_out, c, h*sf, w*sf)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    if isinstance(inputs, list):
                        in_patch = [i[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing] for i in inputs]
                    else:
                        in_patch = inputs[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]

                    if self.opt['datasets'][phase].get('concat_frames_in_batch', False):
                        if isinstance(in_patch, list):
                            in_patch = [i.squeeze(0) for i in in_patch]
                        else:
                            in_patch = in_patch.squeeze(0)
                    
                    self.test(in_patch)
                    out_patch = self.output.detach().cpu()
                    
                    if self.opt['datasets'][phase].get('concat_frames_in_batch', False):
                        out_patch = out_patch.unsqueeze(0)

                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_patch_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_patch_size//2:, :] *= 0
                            out_patch_mask[..., -overlap_patch_size//2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_patch_size//2:] *= 0
                            out_patch_mask[..., :, -overlap_patch_size//2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_patch_size//2, :] *= 0
                            out_patch_mask[..., :overlap_patch_size//2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_patch_size//2] *= 0
                            out_patch_mask[..., :, :overlap_patch_size//2] *= 0

                    E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
            output = E.div_(W)

        else:
            if isinstance(inputs, list):
                _, _, _, h_old, w_old = inputs[0].size()
            else:
                _, _, _, h_old, w_old = inputs.size()
            h_pad = (h_old// window_size[1]+1)*window_size[1] - h_old
            w_pad = (w_old// window_size[2]+1)*window_size[2] - w_old

            if isinstance(inputs, list):
                inputs = [torch.cat([i, i[:,:,:,-h_pad:].flip(3)], 3) for i in inputs]
                inputs = [torch.cat([i, i[:,:,:,:,-w_pad:].flip(4)], 4) for i in inputs]
            else:
                inputs = torch.cat([inputs, inputs[:,:,:,-h_pad:].flip(3)], 3)
                inputs = torch.cat([inputs, inputs[:,:,:,:,-w_pad:].flip(4)], 4)

            if self.opt['datasets'][phase].get('concat_frames_in_batch', False):
                if isinstance(inputs, list):
                    inputs = [i.squeeze(0) for i in inputs]
                else:
                    inputs = inputs.squeeze(0)

            self.test(inputs)
            output = self.output.detach().cpu()

            if self.opt['datasets'][phase].get('concat_frames_in_batch', False):
                output = output.unsqueeze(0)

            output = output[:, :, :, :h_old*sf, :w_old*sf]

        return output