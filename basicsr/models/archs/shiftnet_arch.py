# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is based on the following projects:
# - Shift-Net (https://github.com/dasongli1/Shift-Net)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from einops import rearrange
from typing import Literal
from torch.nn import functional as F

from basicsr.models.archs.tlc_utils import Local_Base
from .op.grouped_spatial_shift import GroupedSpatialShift
from basicsr.models.archs.arch_utils import *


#################### Depth Fusion Blocks ####################
    
    
class ConcatDwConv(nn.Module):
    """Concatenation and Depthwise Convolution"""
    def __init__(self, n_feat_x, n_feat_z, bias=False, **kwargs) -> None:
        super().__init__()
        self.n_feat_x = n_feat_x
        self.n_feat_z = n_feat_z
        self.out_dwconv = nn.Sequential(
            nn.Conv2d(n_feat_x + n_feat_z, n_feat_x, kernel_size=1, bias=bias),
            nn.Conv2d(n_feat_x, n_feat_x, kernel_size=3, stride=1, padding=1, groups=n_feat_x, bias=bias)
        )
    
    def forward(self, input):
        x, z = input[:, 0:self.n_feat_x], input[:, self.n_feat_x:]
        out = self.out_dwconv(torch.cat((x, z), dim=1))
        return torch.cat((out, z), dim=1)


class DAMBlock(nn.Module):
    """Depth-Aware Modulated (DAM) Block.

    Reference:
        Zhu, Qi, et al. "Dast-net: Depth-aware spatio-temporal network for video deblurring."
        2022 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2022.
    """
    def __init__(self, n_feat_x, n_feat_z, scale_bias=1.0, **kwargs):
        super(DAMBlock, self).__init__()
        self.sb = scale_bias
        self.n_feat_x = n_feat_x
        self.n_feat_z = n_feat_z
        self.SFT_scale_conv0 = CDCLayer(n_feat_z, n_feat_x)
        self.SFT_scale_conv1 = CDCLayer(n_feat_x, n_feat_x)
        self.SFT_shift_conv0 = CDCLayer(n_feat_z, n_feat_x)
        self.SFT_shift_conv1 = CDCLayer(n_feat_x, n_feat_x)

    def forward(self, input):
        # x: feature; z: condition
        x, z = input[:, 0:self.n_feat_x], input[:, self.n_feat_x:]
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(z), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(z), 0.1, inplace=True))
        out = x * (scale + self.sb) + shift
        return torch.cat((out, z), dim=1)


class SFTLayer(nn.Module):
    """SFT Layer.

    Modified from: https://github.com/xinntao/SFTGAN/blob/master/pytorch_test/architectures.py
    """
    def __init__(self, n_feat_x, n_feat_z, scale_bias=1.0, **kwargs):
        super(SFTLayer, self).__init__()
        self.n_feat_x = n_feat_x
        self.n_feat_z = n_feat_z
        self.sb = scale_bias
        self.SFT_scale_conv0 = conv3x3(n_feat_z, n_feat_x)
        self.SFT_scale_conv1 = conv3x3(n_feat_x, n_feat_x)
        self.SFT_shift_conv0 = conv3x3(n_feat_z, n_feat_x)
        self.SFT_shift_conv1 = conv3x3(n_feat_x, n_feat_x)

    def forward(self, input):
        # x: feature; z: condition
        x, z = input[:, 0:self.n_feat_x], input[:, self.n_feat_x:]
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(z), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(z), 0.1, inplace=True))
        out = x * (scale + self.sb) + shift
        return torch.cat((out, z), dim=1)
    

class DaTBlock(nn.Module):
    """Depth-aware Transformer (DaT) Block"""
    def __init__(self, n_feat_x, n_feat_z, ffn_expansion_factor=2.66, num_heads=4, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.n_feat_x = n_feat_x
        self.n_feat_z = n_feat_z
        self.norm1 = LayerNorm2d(n_feat_x)
        self.norm2 = LayerNorm2d(n_feat_z)
        self.norm3 = LayerNorm2d(n_feat_x)

        self.cross_attn = CrossAttentionFusionBlock(n_feat_x, n_feat_z, num_heads, bias)
        self.sft_fusion = SFTLayer(n_feat_x, n_feat_x, bias=bias)
        self.ffn = FeedForwardBlock(n_feat_x, ffn_expansion_factor, bias)

        self.beta1 = nn.Parameter(torch.zeros((1, n_feat_x, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, n_feat_x, 1, 1)), requires_grad=True)
    
    def forward(self, input):
        x, z = input[:, 0:self.n_feat_x], input[:, self.n_feat_x:]
        shortcut = x
        z_mod = self.cross_attn(self.norm1(x), self.norm2(z))
        y = self.sft_fusion(torch.cat([shortcut, z_mod], dim=1))[:, 0:self.n_feat_x] *self.beta1 + shortcut
        out = self.ffn(self.norm3(y)) *self.beta2 + shortcut
        return torch.cat([out, z], dim=1)


FUSION_OPTS = Literal['none', 'ConcatConv', "SFT", "DAM", "DaT"]

FUSION_MODULES = {
    'ConcatConv': ConcatDwConv,
    'DAM': DAMBlock,
    'SFT': SFTLayer,
    'DaT': DaTBlock,
}

class DepthFusionBlock(nn.Module):
    def __init__(self, 
                 n_feat_x, 
                 n_feat_z, 
                 num_depth_fusion_blocks=2, 
                 fusion_opt:FUSION_OPTS='DaT', 
                 shift_depth_feat=False,
                 s_factor=4, 
                 **kwargs):
        super(DepthFusionBlock, self).__init__()
        self.fusion_opt = fusion_opt
        self.n_feat_x = n_feat_x
        self.n_feat_z = n_feat_z
        self.num_depth_fusion_blocks = num_depth_fusion_blocks
        self.shift_depth_feat = shift_depth_feat
        self.s_factor = s_factor

        self.fusion = [FUSION_MODULES[self.fusion_opt](n_feat_x, n_feat_z, **kwargs) for _ in range(num_depth_fusion_blocks)]
        self.fusion = nn.Sequential(*self.fusion)

        # Spatial shift for depth features
        if not shift_depth_feat:
            return
        n1, n2 = self.check_num_multiframe_features(n_feat_z)
        shifts_x, shifts_y = self.get_shifts(n1, n2, s_factor)
        self.spatial_shift = GroupedSpatialShift(shifts_x, shifts_y)
        self.agg_conv = nn.Conv2d(n_feat_z, n_feat_z, kernel_size=3, padding=1, groups=n_feat_z, bias=False)

    def forward(self, x, z):
        if self.shift_depth_feat:
            z = self.spatial_shift(z)
            z = self.agg_conv(z)
        input = torch.cat([x, z], dim=1)
        out = self.fusion(input)
        return out[:, 0:self.n_feat_x]
    
    def check_num_multiframe_features(self, num_feat, max_value=10):
        """Check number of multi-frame features and get n1, n2 for spatial shift"""
        assert num_feat / 8 > 0, "Number of multi-frame features must be greater than 8."
        for n1 in range(1, max_value+1):
            for n2 in range(0, max_value+1):
                if num_feat % (n1*8 + n2*16) == 0:
                    return n1, n2
        raise ValueError(f"Invalid number of multi-frame features: {num_feat}. Must be divisible by n1*8 + n2*16.")
    
    def get_shifts(self, n1, n2, s_factor=4):
        """Get shifts for spatial shift"""
        directions1 = [
            (1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)
        ]
        directions2 = [
            (2, 2), (2, 1), (2, 0), (2, -1), (2, -2),
            (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
            (1, 2), (1, -2), (0, 2), (0, -2), (-1, 2), (-1, -2)
        ]
        shifts = [(dx * s_factor, dy * s_factor) for _ in range(n2) for dx, dy in directions2]
        shifts.extend([(dx * s_factor, dy * s_factor) for _ in range(n1) for dx, dy in directions1])
        shifts_x, shifts_y = zip(*shifts)
        shifts_x_tensor = torch.tensor(shifts_x, dtype=torch.int32)
        shifts_y_tensor = torch.tensor(shifts_y, dtype=torch.int32)
        return shifts_x_tensor, shifts_y_tensor
        

#################### Spatio-Temporal Shift Blocks and Multi-Frame Fusion ####################

 
class GroupedSpatioTemporalShiftBlock(nn.Module):
    """Grouped Spatio-Temporal Shift Block"""
    def __init__(self, n_feat, shifts_x, shifts_y, reverse:bool, div=2, bias=False):
        super().__init__()
        self.n_feat = n_feat
        self.shift_feat = int(n_feat) // div
        self.reverse = reverse
        self.conv1 = nn.Conv2d(self.shift_feat, self.shift_feat, 3, bias=bias, padding=1, groups=self.shift_feat)
        self.spatial_shift = GroupedSpatialShift(shifts_x, shifts_y)

    def forward(self, x):
        B, N, C, H, W = x.shape
        
        # Temporal shift
        if N > 1:
            x = x.view(B, N*C, H, W)
            shift = -self.shift_feat if self.reverse else self.shift_feat
            x = torch.roll(x, shift, 1)

        # Spatial shift
        x = x.view(B*N, C, H, W)
        x_ss = x[:, 0:self.shift_feat] if not self.reverse else x[:, -self.shift_feat:]
        x_ss = self.spatial_shift(x_ss)
        x_ss = self.conv1(x_ss)
        out = torch.cat((x, x_ss), dim=1)

        return out.view(B, N, -1, H, W)
    

class LightweightFeatureFusion(nn.Module):
    """Lightweight Feature Fusion Block"""
    def __init__(self, n_feat, add_feat=0, bias=False):
        super(LightweightFeatureFusion, self).__init__()

        self.n_feat = n_feat
        self.add_feat = add_feat

        # conv layers
        self.conv1x1_1 = nn.Conv2d(n_feat + add_feat, 2*n_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        self.dw_conv3x3 = nn.Conv2d(2*n_feat, 2*n_feat, kernel_size=3, padding=1, stride=1, groups=2*n_feat, bias=bias)
        # self.resblock_1 = ResBlock(2*n_feat, 7, bias)
        self.conv1x1_2 = nn.Conv2d(n_feat, 2*n_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        self.conv1x1_3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

        # parallel convolution block
        self.sk_layer = ParallelConvBlock(n_feat, bias=bias, large_kernel_size=7)

        # gates
        self.sg1 = SimpleGate()
        self.sg2 = SigmoidGate()

        # channel attention
        self.ca = CALayer(n_feat, bias=bias, reduction=4)

        # normalization layer
        self.norm1 = LayerNorm2d(n_feat + add_feat)

        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        shortcut = x[:, 0:self.n_feat]
        x = self.norm1(x)
        x = self.conv1x1_1(x)
        x = self.dw_conv3x3(x)
        # x = self.resblock_1(x)
        x = self.sg1(x)
        x = self.sk_layer(x)
        x = self.conv1x1_2(x)
        x = self.sg2(x)
        x = self.ca(x)
        x = self.conv1x1_3(x)

        y = shortcut + x * self.beta

        return rearrange(y, '(b n) c h w -> b n c h w', n=N)


class GroupedSpatioTemporalShiftFusion(nn.Module):
    """Grouped Spatio-Temporal Shift Fusion"""
    def __init__(self, num_multiframe_features, 
                 num_gstshift_blocks=4,
                 num_lightweight_blocks=2,
                 bias=False, 
                 s_factor=4,
                 **kwargs):
        super(GroupedSpatioTemporalShiftFusion, self).__init__()
        # Check number of multi-frame features and get n1, n2
        n1, n2 = self.check_num_multiframe_features(num_multiframe_features//2)

        self.num_multiframe_features = num_multiframe_features
        self.s_factor = s_factor

        # Get shifts for spatial shift
        shifts_x, shifts_y = self.get_shifts(n1, n2, s_factor=s_factor)

        # Shift and fusion blocks
        self.body = []
        for i in range(num_gstshift_blocks):
            reverse = i % 2 == 1 # reverse every other block
            self.body.append(GroupedSpatioTemporalShiftBlock(num_multiframe_features, 
                                                             shifts_x, 
                                                             shifts_y, 
                                                             reverse=reverse, 
                                                             bias=bias))
            self.body.append(self.add_lightweight_fusion_block(num_multiframe_features, 
                                                               add_feat=num_multiframe_features//2, 
                                                               num_blocks=num_lightweight_blocks, 
                                                               bias=bias))

        self.body = nn.Sequential(*self.body)

    def add_lightweight_fusion_block(self, n_feat, add_feat, num_blocks, bias):
        """Add lightweight fusion blocks"""
        assert num_blocks > 0
        fusion_blocks = [LightweightFeatureFusion(n_feat, add_feat, bias)]
        if num_blocks > 1:
            fusion_blocks += [LightweightFeatureFusion(n_feat, bias) for _ in range(num_blocks-1)]
        return nn.Sequential(*fusion_blocks)
    
    def check_num_multiframe_features(self, num_feat, max_value=10):
        """Check number of multi-frame features and get n1, n2 for spatial shift"""
        for n1 in range(1, max_value+1):
            for n2 in range(1, max_value+1):
                if num_feat % (n1*8 + n2*16) == 0:
                    return n1, n2
        raise ValueError(f"Invalid number of multi-frame features: {num_feat}. Must be divisible by n1*8 + n2*16.")
    
    def get_shifts(self, n1, n2, s_factor=4):
        """Get shifts for spatial shift"""
        directions1 = [
            (1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)
        ]
        directions2 = [
            (2, 2), (2, 1), (2, 0), (2, -1), (2, -2),
            (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
            (1, 2), (1, -2), (0, 2), (0, -2), (-1, 2), (-1, -2)
        ]
        shifts = [(dx * s_factor, dy * s_factor) for _ in range(n2) for dx, dy in directions2]
        shifts.extend([(dx * s_factor, dy * s_factor) for _ in range(n1) for dx, dy in directions1])
        shifts_x, shifts_y = zip(*shifts)
        shifts_x_tensor = torch.tensor(shifts_x, dtype=torch.int32)
        shifts_y_tensor = torch.tensor(shifts_y, dtype=torch.int32)
        return shifts_x_tensor, shifts_y_tensor
    

    def forward(self, x):
        return self.body(x)


class MultiFrameFeatureFusion(nn.Module):
    """Multi-Frame Feature Fusion"""
    def __init__(self, 
                 num_rgb_processing_features, 
                 num_rgb_multiframe_features,
                 num_rgb_mf_encoding_blocks=3,
                 is_depth_fusion=False,
                 shift_depth_feat=False,
                 num_depth_processing_features=16,
                 num_depth_multiframe_features=48,
                 num_depth_mf_encoding_blocks=1,
                 num_depth_fusion_blocks=2,
                 kernel_size=3, 
                 reduction=4, 
                 bias=False,
                 **kwargs):
        super(MultiFrameFeatureFusion, self).__init__()
        act = nn.PReLU()
        self.act = act
        self.shift_depth_feat = shift_depth_feat

        # Input Convolution
        self.in_conv = CAB(num_rgb_processing_features,
                            kernel_size=kernel_size,
                            reduction=reduction,
                            act=act,
                            bias=bias,
                            **kwargs)
        
        
        # Encoder layers
        self.x_encoder_level1 = [
            GroupedSpatioTemporalShiftFusion(
                num_rgb_multiframe_features,
                bias=bias,
                **kwargs
                ) for _ in range(num_rgb_mf_encoding_blocks)]
        self.x_encoder_level1 = nn.Sequential(*self.x_encoder_level1)
        
        self.x_encoder_level2 = [
            GroupedSpatioTemporalShiftFusion(
                num_rgb_multiframe_features,
                bias=bias,
                **kwargs
                ) for _ in range(num_rgb_mf_encoding_blocks)]
        self.x_encoder_level2 = nn.Sequential(*self.x_encoder_level2)
        
        # Downsample layers
        self.x_down01 = DownSample(num_rgb_processing_features, num_rgb_multiframe_features - num_rgb_processing_features)
        self.x_down12 = DownSample(num_rgb_multiframe_features, 0)

        # Decoder layers
        self.x_decoder_level1 = [
            GroupedSpatioTemporalShiftFusion(
                num_rgb_multiframe_features,
                bias=bias,
                **kwargs
                ) for _ in range(num_rgb_mf_encoding_blocks)]
        self.x_decoder_level1 = nn.Sequential(*self.x_decoder_level1)

        self.x_decoder_level2 = [
            GroupedSpatioTemporalShiftFusion(
                num_rgb_multiframe_features,
                bias=bias,
                **kwargs
                ) for _ in range(num_rgb_mf_encoding_blocks)]
        self.x_decoder_level2 = nn.Sequential(*self.x_decoder_level2)

        # Skip Attention layers
        self.x_skip_attn0 = CAB(num_rgb_processing_features,
                                kernel_size=kernel_size,
                                reduction=reduction,
                                act=act,
                                bias=bias,
                                **kwargs)
        self.x_skip_attn1 = CAB(num_rgb_multiframe_features, 
                                kernel_size=kernel_size,
                                reduction=reduction, 
                                act=act,
                                bias=bias,
                                **kwargs)
        
        # Upsample layers
        self.x_up21 = SkipUpSample(num_rgb_multiframe_features, 0)
        self.x_up10 = nn.Sequential(
            PixelShufflePack(num_rgb_multiframe_features, num_rgb_processing_features, 2, upsample_kernel=3),
            act,
            conv(num_rgb_processing_features, num_rgb_processing_features, kernel_size, bias=bias)
        )

        # Output Convolution
        self.out_conv = CAB(num_rgb_processing_features,
                            kernel_size=kernel_size,
                            reduction=reduction,
                            act=act,
                            bias=bias,
                            **kwargs)
        
        if not is_depth_fusion:
            return
        
        # Depth Encoder layers
        if shift_depth_feat:
            self.z_encoder_level1 = [
                GroupedSpatioTemporalShiftFusion(
                    num_depth_multiframe_features,
                    num_gstshift_blocks=2,
                    num_lightweight_blocks=1,
                    bias=bias,
                    **kwargs
                    ) for _ in range(num_depth_mf_encoding_blocks)]
            self.z_encoder_level1 = nn.Sequential(*self.z_encoder_level1)

            self.z_encoder_level2 = [
                GroupedSpatioTemporalShiftFusion(
                    num_depth_multiframe_features,
                    num_gstshift_blocks=2,
                    num_lightweight_blocks=1,
                    bias=bias,
                    **kwargs
                    ) for _ in range(num_depth_mf_encoding_blocks)]
            self.z_encoder_level2 = nn.Sequential(*self.z_encoder_level2)

        # Depth Downsample layers
        self.z_down01 = DownSample(num_depth_processing_features, num_depth_multiframe_features - num_depth_processing_features)
        self.z_down12 = DownSample(num_depth_multiframe_features, 0)

        # Depth Fusion layers
        self.xz_fusion_level1 = DepthFusionBlock(
            n_feat_x=num_rgb_multiframe_features,
            n_feat_z=num_depth_multiframe_features,
            num_depth_fusion_blocks=num_depth_fusion_blocks,
            shift_depth_feat=False,
            bias=bias,
            **kwargs)
        
        self.xz_fusion_level2 = DepthFusionBlock(
            n_feat_x=num_rgb_multiframe_features,
            n_feat_z=num_depth_multiframe_features,
            num_depth_fusion_blocks=num_depth_fusion_blocks,
            shift_depth_feat=False,
            bias=bias,
            **kwargs)

    def forward(self, x, z=None):
        B, N, Cx, H, W = x.shape
        x = x.view(B*N, Cx, H, W)
        x = self.in_conv(x)
        shortcut = x 
        x = self.x_down01(x)

        enc1 = self.x_encoder_level1(rearrange(x, '(b n) c h w -> b n c h w', n=N))
        enc1 = rearrange(enc1, 'b n c h w -> (b n) c h w')
        enc1_down = self.x_down12(enc1)
        enc2 = self.x_encoder_level2(rearrange(enc1_down, '(b n) c h w -> b n c h w', n=N))

        if z is not None:
            z = self.z_down01(rearrange(z, 'b n c h w -> (b n) c h w', n=N))
            z = rearrange(z, '(b n) c h w -> b n c h w', n=N)
            enc1_z = self.z_encoder_level1(z) if self.shift_depth_feat else z
            enc1_z = rearrange(enc1_z, 'b n c h w -> (b n) c h w')
            enc1_down_z = self.z_down12(enc1_z)
            enc1_down_z = rearrange(enc1_down_z, '(b n) c h w -> b n c h w', n=N)
            enc2_z = self.z_encoder_level2(enc1_down_z) if self.shift_depth_feat else enc1_down_z
            enc2_z = rearrange(enc2_z, 'b n c h w -> (b n) c h w')

        dec2 = self.x_decoder_level2(enc2)
        dec2 = rearrange(dec2, 'b n c h w -> (b n) c h w')
        if z is not None:
            dec2 = self.xz_fusion_level2(dec2, enc2_z)

        x = self.x_up21(dec2, self.x_skip_attn1(enc1))
        dec1 = self.x_decoder_level1(rearrange(x, '(b n) c h w -> b n c h w', n=N))
        dec1 = rearrange(dec1, 'b n c h w -> (b n) c h w')
        if z is not None:
            dec1 = self.xz_fusion_level1(dec1, enc1_z)

        x = self.x_up10(dec1) + self.x_skip_attn0(shortcut)
        y = self.out_conv(x)
        return rearrange(y, '(b n) c h w -> b n c h w', n=N)


#################### Deep Feature extractor ####################


BASE_BLOCKS = Literal["CAB", "DwConv"]


class FeatExt_Unet(nn.Module):
    """U-Net Feature Extraction"""
    def __init__(self, n_feat, add_feat, act, kernel_size=3, bias=False, base_block:BASE_BLOCKS='CAB', **kwargs):
        super(FeatExt_Unet, self).__init__()
        base_module = {
            'DwConv': DwConvBlock,
            'CAB': CAB,
        }[base_block]

        self.encoder_l1 = [base_module(n_feat, kernel_size, act=act, bias=bias, **kwargs) for _ in range(1)]
        self.encoder_l2 = [base_module(n_feat + add_feat, kernel_size, act=act, bias=bias, **kwargs) for _ in range(3)]
        self.encoder_l3 = [base_module(n_feat + 2*add_feat, kernel_size, act=act, bias=bias, **kwargs) for _ in range(3)]
        self.encoder_l1 = nn.Sequential(*self.encoder_l1)
        self.encoder_l2 = nn.Sequential(*self.encoder_l2)
        self.encoder_l3 = nn.Sequential(*self.encoder_l3)
        self.down12 = DownSample(n_feat, add_feat)
        self.down23 = DownSample(n_feat+add_feat, add_feat)

        self.decoder_l1 = [base_module(n_feat, kernel_size, act=act, bias=bias, **kwargs) for _ in range(1)]
        self.decoder_l2 = [base_module(n_feat+add_feat, kernel_size, act=act, bias=bias, **kwargs) for _ in range(3)]
        self.decoder_l3 = [base_module(n_feat+add_feat*2, kernel_size, act=act, bias=bias, **kwargs) for _ in range(3)]
        self.decoder_l1 = nn.Sequential(*self.decoder_l1)
        self.decoder_l2 = nn.Sequential(*self.decoder_l2)
        self.decoder_l3 = nn.Sequential(*self.decoder_l3)

        self.up21 = SkipUpSample(n_feat, add_feat)
        self.up32 = SkipUpSample(n_feat + add_feat, add_feat)

    def forward(self, x):
        # encoder
        enc1 = self.encoder_l1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_l2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_l3(x)

        # decoder
        dec3 = self.decoder_l3(enc3)
        x = self.up32(dec3, enc2)
        dec2 = self.decoder_l2(x)
        x = self.up21(dec2, enc1)
        dec1 = self.decoder_l1(x)
        return dec1
        

#################### ShiftNet architecture ####################


class ShiftNetPlus(nn.Module):
    """
    ShiftNet architecture.
    
    This class implements the baseline (RGB only) ShiftNet architecture and the extended RGBD ShiftNet architecture.
    If `aux_data` contains 'depth', 'mono_depth_sharp', or 'mono_depth_blur', the RGBD ShiftNet architecture is used.

    Args:
        aux_data (list): List of auxiliary data types. Options: ['depth', 'mono_depth_sharp', 'mono_depth_blur'].
        input_channels (int): Number of input channels. Default: 3.
        num_rgb_processing_features (int): Number of RGB processing features. Default: 16.
        num_rgb_multiframe_features (int): Number of RGB multi-frame features. Default: 64.
        num_depth_processing_features (int): Number of depth processing features. Default: 16.
        num_depth_multiframe_features (int): Number of depth multi-frame features. Default: 48.
        stack_size_rgb_feat_ext (int): Number of RGB feature extraction blocks. Default: 3.
        stack_size_depth_feat_ext (int): Number of depth feature extraction blocks. Default: 1.
        stack_size_restoration (int): Number of restoration blocks. Default: 3.
        depth_feat_base_block (BASE_BLOCKS): Base block for depth feature extraction. Default: 'CAB'.
        rgb_feat_base_block (BASE_BLOCKS): Base block for RGB feature extraction. Default: 'CAB'.
        shift_depth_feat (bool): Shift depth features. Default: True.
        fusion_opt (FUSION_OPTS): Fusion option. Default: 'DaT'.
        out_frames (int): Number of output frames. Default: 1.
        model_stride (int): Model stride. Default: 1.
        kernel_size (int): Kernel size. Default: 3.
        reduction (int): Reduction factor. Default: 4.
        bias (bool): Use bias. Default: False.

    Reference:
        Li, Dasong, et al. "A simple baseline for video restoration with grouped spatial-temporal shift." 
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    
    Original implementation: https://github.com/dasongli1/Shift-Net
    
    """
    def __init__(self,
                 aux_data,
                 input_channels=3,
                 num_rgb_processing_features=16,
                 num_rgb_multiframe_features=64,
                 num_depth_processing_features=16,
                 num_depth_multiframe_features=48,
                 stack_size_rgb_feat_ext=2,
                 stack_size_depth_feat_ext=1,
                 stack_size_restoration=2,
                 depth_feat_base_block='CAB',
                 rgb_feat_base_block='CAB',
                 shift_depth_feat=True,
                 fusion_opt='DaT',
                 out_frames=1,
                 model_stride=1,
                 kernel_size=3,
                 reduction=4,
                 bias=False,
                 **kwargs
                 ):
        super(ShiftNetPlus, self).__init__()
        self.num_rgb_processing_features = num_rgb_processing_features
        self.num_rgb_multiframe_features = num_rgb_multiframe_features
        self.num_depth_processing_features = num_depth_processing_features
        self.num_depth_multiframe_features = num_depth_multiframe_features
        self.out_frames = out_frames
        self.model_stride = model_stride
        self.shift_depth_feat = shift_depth_feat
        self.stack_size_rgb_feat_ext = stack_size_rgb_feat_ext
        self.stack_size_depth_feat_ext = stack_size_depth_feat_ext
        self.stack_size_restoration = stack_size_restoration
        self.aux_data = aux_data
        self.fusion_opt = fusion_opt if any(element in ['depth', 'mono_depth_sharp', 'mono_depth_blur'] for element in self.aux_data) else 'none'
        self.depth_feat_base_block = depth_feat_base_block
        self.rgb_feat_base_block = rgb_feat_base_block
        self.is_depth_fusion = any(element in ['depth', 'mono_depth_sharp', 'mono_depth_blur'] for element in self.aux_data)
        self.bias = bias
        self.reduction = reduction
        self.kernel_size = kernel_size


        # Shallow Image feature extraction
        base_module = self._get_base_rgb_feat_block()
        self.shallow_rgb_feat_extract = nn.Sequential(
            nn.Conv2d(input_channels, self.num_rgb_processing_features, 7, 1, 3),
            base_module(self.num_rgb_processing_features, kernel_size=5, bias=bias, act=nn.PReLU(), reduction=reduction)
            )

        # Deep Image feature extraction
        self.deep_rgb_extract = [
            FeatExt_Unet(
                self.num_rgb_processing_features, 
                add_feat=self.num_rgb_processing_features//4,
                kernel_size=self.kernel_size,
                base_block=self.rgb_feat_base_block,
                reduction=self.reduction, 
                act=nn.PReLU(),
                bias=self.bias, 
                ) for _ in range(self.stack_size_rgb_feat_ext)
            ]
        self.deep_rgb_extract = nn.Sequential(*self.deep_rgb_extract)
        
        # Multi-frame feature fusion
        self.multi_frame_fusion = MultiFrameFeatureFusion(
            num_rgb_processing_features=self.num_rgb_processing_features,
            num_rgb_multiframe_features=self.num_rgb_multiframe_features,
            is_depth_fusion=self.is_depth_fusion,
            shift_depth_feat=self.shift_depth_feat,
            num_depth_processing_features=self.num_depth_processing_features,
            num_depth_multiframe_features=self.num_depth_multiframe_features,
            fusion_opt=self.fusion_opt,
            reduction=self.reduction,
            kernel_size=self.kernel_size,
            bias=self.bias,
            **kwargs
        )
        
        # Image feature restoration
        self.s3_catconv = conv(self.num_rgb_processing_features*3, self.num_rgb_processing_features, 1, bias=True)
        self.deep_restoration = [
            FeatExt_Unet(
                self.num_rgb_processing_features, 
                add_feat=self.num_rgb_processing_features//4,
                kernel_size=self.kernel_size,
                base_block=self.rgb_feat_base_block,
                reduction=self.reduction, 
                act=nn.PReLU(),
                bias=self.bias, 
                ) for _ in range(self.stack_size_restoration)]
        self.deep_restoration = nn.Sequential(*self.deep_restoration)
        
        # Final output convolution
        self.out_conv = conv(self.num_rgb_processing_features, 3, 3, bias=False) 
        
        # Skip depth block if no depth data
        #if any(element not in ['depth', 'mono_depth_sharp', 'mono_depth_blur'] for element in self.aux_data):
        if not self.is_depth_fusion:
            return

        ############################################################
        # Depth blocks

        # Shallow Depth feature extraction
        base_module = self._get_base_depth_feat_block()
        self.shallow_depth_feat_extract = nn.Sequential(nn.Conv2d(1, self.num_depth_processing_features, 7, 1, 3),
            base_module(self.num_depth_processing_features, 5, bias=False, act=nn.PReLU(), reduction=self.reduction)
        )

        # Deep Depth feature extraction
        self.deep_depth_extract = [
            FeatExt_Unet(
                self.num_depth_processing_features,
                add_feat=self.num_depth_processing_features//4,
                kernel_size=self.kernel_size,
                base_block=self.depth_feat_base_block,
                reduction=self.reduction,
                act=nn.PReLU(),
                bias=self.bias,
                ) for _ in range(self.stack_size_depth_feat_ext)
            ]
        self.deep_depth_extract = nn.Sequential(*self.deep_depth_extract)

        # Depth feature fusion
        self.depth_fusion_s0 = DepthFusionBlock(
            n_feat_x=self.num_rgb_processing_features,
            n_feat_z=self.num_depth_processing_features,
            fusion_opt=self.fusion_opt,
            shift_depth_feat=self.shift_depth_feat,
            bias=self.bias,
            **kwargs)
        
        self.depth_fusion_s1 = DepthFusionBlock(
            n_feat_x=self.num_rgb_processing_features,
            n_feat_z=self.num_depth_processing_features,
            fusion_opt=self.fusion_opt,
            shift_depth_feat=self.shift_depth_feat,
            bias=self.bias,
            **kwargs)

            
    def _get_base_rgb_feat_block(self):
        """Get base block for RGB feature extraction"""
        base_block = {
            'DwConv': DwConvBlock,
            'CAB': CAB,
        }[self.rgb_feat_base_block]
        return base_block
    

    def _get_base_depth_feat_block(self):
        """Get base block for depth feature extraction"""
        base_block = {
            'DwConv': DwConvBlock,
            'CAB': CAB,
        }[self.depth_feat_base_block]
        return base_block
    
    def _get_inputs(self, inputs):
        """Get inputs (x, z) from the input list."""
        assert type(inputs) == list
        if len(inputs) == 2 and any(element in ['depth', 'mono_depth_sharp', 'mono_depth_blur'] for element in self.aux_data):
            x = inputs[0]
            z = inputs[1]
            return x, z
        elif len(inputs) == 2 and inputs[1] is None:
            x = inputs[0]
            return x, None
        elif len(inputs) == 1:
            x = inputs[0]
            return x, None
    
    def stage0(self, x, z):
        """Stage 0: shallow feature extraction"""
        B, N, Cx, H, W = x.shape
        # Shallow feature extraction (RGB)
        x0 = self.shallow_rgb_feat_extract(x.view(B*N, Cx, H, W))

        # Skip depth feature extraction if no depth data
        if z is not None:
            Cz = z.shape[2]
            # Shallow feature extraction (Depth)
            z0 = self.shallow_depth_feat_extract(z.view(B*N, Cz, H, W))
            x0 = self.depth_fusion_s0(x0, z0)   
            return x0.view(B, N, -1, H, W), z0.view(B, N, -1, H, W)
        else:
            return x0.view(B, N, -1, H, W), None

    def stage1(self, x0, z0):
        """Stage 1: deep feature extraction and depth feature fusion"""
        B, N, Cx, H, W = x0.shape
        x0 = x0.view(B*N, Cx, H, W)
        # Deep feature extraction (RGB)
        x1 = x0 + self.deep_rgb_extract(x0)

        # Skip depth feature extraction and fusion if no depth data
        if z0 is not None:
            Cz = z0.shape[2]
            z0 = z0.view(B*N, Cz, H, W)
            # Deep feature extraction (Depth)
            z1 = z0 + self.deep_depth_extract(z0)

            # Depth feature fusion
            x1 = self.depth_fusion_s1(x1, z1)
            return x1.view(B, N, -1, H, W), z1.view(B, N, -1, H, W)
        else:
            return x1.view(B, N, -1, H, W), None
        
    def stage2(self, x1, z1):
        """Stage 2: multi-frame feature fusion and depth feature fusion"""
        # Multi-frame feature fusion
        x2 = self.multi_frame_fusion(x1, z1)
        return x2
    
    def stage3(self, x0, x1, x2):
        """Stage 3: image feature restoration"""
        B, N, Cx, H, W = x0.shape
        x0 = x0.view(B*N, -1, H, W)
        x1 = x1.view(B*N, -1, H, W)
        x2 = x2.view(B*N, -1, H, W)

        x = self.s3_catconv(torch.cat((x0, x1, x2), dim=1))
        shortcut = x 
        x = self.deep_restoration(x)
        x = self.out_conv(x + shortcut)
        return x

    def forward(self, inputs):
        # Get inputs
        x, z = self._get_inputs(inputs)
        B, N, C, H, W = x.shape
        shortcut = x

        # Stage 0: shallow feature extraction
        x0, z0 = self.stage0(x, z)

        # Stage 1: deep feature extraction and depth feature fusion
        x1, z1 = self.stage1(x0, z0)

        # Stage 2: multi-frame feature fusion and depth feature fusion
        x2 = self.stage2(x1, z1)

        # Stage 3: image feature restoration
        if N > 1:
            stencil = self.model_stride if self.model_stride else (N - self.out_frames) // 2
            output_features = self.stage3(x0[:,stencil:-stencil], x1[:,stencil:-stencil], x2[:,stencil:-stencil])
            prediction = output_features + shortcut[:,stencil:-stencil].view(-1, C, H, W)
        else:
            output_features = self.stage3(x0, x1, x2)
            prediction = output_features + shortcut.view(-1, C, H, W)
        return prediction.view(B, -1, C, H, W)


#################### Local version of ShiftNet (TLC) ####################


class LocalShiftNetPlus(Local_Base, ShiftNetPlus):
    """
    Local-ShiftNet architecture.

    This class implements the utility functions for using the Test-time Local Converter (TLC) with the ShiftNet architecture.
    """
    def __init__(self, *args, train_size=(1, 3, 3, 256, 256), base_size=None, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ShiftNetPlus.__init__(self, *args, **kwargs)
        B, N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))
        self.register_buffer('dummy_imgs', torch.randn(train_size))
        self.register_buffer('dummy_depths', torch.randn((B, N, 1, H, W)))
        self.train_size = train_size
        self.base_size = base_size
        self.fast_imp = fast_imp
        self.kwargs = kwargs

        self.eval()

    def init_TLC(self, tlc_layers):
        inputs = [self.dummy_imgs]
        if any(element in ['depth', 'mono_depth_sharp', 'mono_depth_blur'] for element in self.kwargs.get('aux_data', None)):
            inputs.append(self.dummy_depths)
        with torch.no_grad():
            self.convert(base_size=self.base_size,
                         inputs=inputs,
                         train_size=self.train_size, 
                         fast_imp=self.fast_imp,
                         tlc_layers=tlc_layers, 
                         **self.kwargs)
        
        # clean dummy data
        del self.dummy_imgs
        del self.dummy_depths
