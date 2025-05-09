# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# - BasicSR (https://github.com/XPixelGroup/BasicSR)
# - HINet (https://github.com/megvii-model/HINet)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torchvision
from distutils.version import LooseVersion
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from einops import rearrange
from torch.nn import functional as F

from basicsr.models.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


#################### EDVR modules ####################

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
    

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


#################### ShiftNet modules ####################


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
  

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


class LayerNormFunction(torch.autograd.Function):
    """Layer Normalization Function (LayerNorm)"""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    """Layer Normalization Layer for 2D data"""
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CALayer(nn.Module):
    """Channel Attention Layer (CALayer)"""
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        reduction = 1
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SCALayer(nn.Module):
    """Simplified Channel Attention Layer (SCALayer)"""
    def __init__(self, channel, bias=True):
        super(SCALayer, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
        )

    def forward(self, x):
        return  x * self.sca(x)


class CAB(nn.Module):
    """Channel Attention Block (CAB)"""
    def __init__(self, n_feat, kernel_size, reduction, bias, act, **kwargs):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class SimpleGate(nn.Module):
    """Simple Gate"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SigmoidGate(nn.Module):
    """Sigmoid Gate"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)
    

class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, n_feat, kernel_size, bias):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat)
    def forward(self, x):
        res = self.conv(x)
        return res + x


class ParallelConvBlock(nn.Module):
    """Parallel Super Kernel Block"""
    def __init__(self, n_feat, large_kernel_size=5, bias=False):
        super(ParallelConvBlock, self).__init__()
        self.conv_large = nn.Conv2d(n_feat, n_feat, large_kernel_size, bias=bias, padding=large_kernel_size//2, groups=n_feat)
        self.conv_small = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat)
    def forward(self, x):
        res_1 = self.conv_large(x)
        # return res_1
        res_2 = self.conv_small(x)
        return res_1 + res_2 + x
    

class PixelShufflePack(nn.Module):
    """Pixel Shuffle layer."""
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
    padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x    


class DownSample(nn.Module):
    """Downsample layer."""
    def __init__(self, in_channels, add_channels):
        super(DownSample, self).__init__()
        # self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
        #                           nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
        # self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True),
        #                    nn.PReLU())
        self.down = nn.Conv2d(in_channels, in_channels + add_channels, kernel_size=3, stride=2, padding=1, bias=True)
    def forward(self, x):
        x = self.down(x)
        return x


class SkipUpSample(nn.Module):
    """Skip and upsample layer."""
    def __init__(self, in_channels, add_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(in_channels + add_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class UpSample(nn.Module):
    """Upsample layer."""
    def __init__(self, in_channels, add_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + add_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
    

class CDCLayer(nn.Module):
    """Center Difference Convolution (CDC) Layer.

    Modified from: https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/models/CDCNs.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.8):

        super(CDCLayer, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class DwConvBlock(nn.Module):
    """Depthwise Convolution Block"""
    def __init__(self, n_feat, kernel_size, act, bias):
        super(DwConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_feat, bias=bias),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=bias),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_feat, bias=bias),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=bias),
            act
        )

    def forward(self, x):
        return self.conv_block(x)


class CrossAttentionFusionBlock(nn.Module):
    """Multi Head Transposed Cross-Attention Block.
    
    Inspired by the Multi-Dconv Head Transposed Attention (MDHTA) module used in the Restormer architecture.
    
    Reference:
        Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for high-resolution image restoration." 
        Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
    """
    def __init__(self, n_feat_x, n_feat_z, num_heads=4, bias=False):
        super(CrossAttentionFusionBlock, self).__init__()
        self.n_feat_x = n_feat_x
        self.n_feat_z = n_feat_z
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv_dwconv = nn.Sequential(
            nn.Conv2d(n_feat_z, n_feat_z*2, kernel_size=1, bias=bias),
            nn.Conv2d(n_feat_z*2, n_feat_z*2, kernel_size=3, stride=1, padding=1, groups=n_feat_z, bias=bias)
        )
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(n_feat_x, n_feat_x, kernel_size=1, bias=bias),
            nn.Conv2d(n_feat_x, n_feat_x, kernel_size=3, stride=1, padding=1, groups=n_feat_x, bias=bias)
        )
        self.project_out = nn.Conv2d(n_feat_x, n_feat_x, kernel_size=1, bias=bias)

    def forward(self, x, z):
        b,cx,h,w = x.shape
        cz = z.shape[1]

        kv = self.kv_dwconv(z)
        q = self.q_dwconv(x)

        k,v = kv.chunk(2, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class FeedForwardBlock(nn.Module):
    """Gated-Dconv Feed Forward network.
    
    Borrowed from the restormer architecture (https://github.com/swz30/Restormer).

    Reference:
        Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for high-resolution image restoration." 
        Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
    """
    def __init__(self, n_feat, ffn_expansion_factor=2.66, bias=False):
        super(FeedForwardBlock, self).__init__()

        hidden_features = int(n_feat*ffn_expansion_factor)

        self.project_in = nn.Conv2d(n_feat, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

