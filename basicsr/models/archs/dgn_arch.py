# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code implements the Depth Guided Network (DGN) architecture.
#
# Reference:
#     Li, Lerenhan, et al. "Dynamic scene deblurring by depth guided model." 
# #   IEEE Transactions on Image Processing 29 (2020): 5273-5288.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        residual = x        # Store input for the residual connection
        out = self.layers(x)
        out += residual     # Adding the input (residual connection)

        return out
    

class SFTLayer(nn.Module):
    """SFT Layer
    Modified from: https://github.com/xinntao/SFTGAN/blob/master/pytorch_test/architectures.py
    """
    def __init__(self, feature_channels, condition_channels=1):
        super(SFTLayer, self).__init__()
        self.condition_conv_block = nn.Sequential(
            nn.Conv2d(condition_channels, feature_channels, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.SFT_scale_conv_block = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.SFT_shift_conv_block = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, z):
        # x: feature; z: condition
        z = self.condition_conv_block(z)
        scale = self.SFT_scale_conv_block(z)
        shift = self.SFT_shift_conv_block(z)
        return x * scale + shift
    

class DGN(nn.Module):
    """DGN network structure.
    
    Args:
        aux_data (list): A list of auxiliary data. If not None, the network will use the auxiliary data to condition the network.
        num_processing_features (int): Number of processing features in the network.
        input_channels (int): Number of input channels.
        condition_channels (int): Number of condition channels.
        out_frames (int): Number of output frames.
    """
    def __init__(self, aux_data, num_processing_features=32, input_channels=3, condition_channels=1, out_frames=1, **kwargs):
        super(DGN, self).__init__()
        self.aux_data = aux_data
        self.out_frames = out_frames
        
        self.conv1 = nn.Conv2d(input_channels, num_processing_features, kernel_size=7, stride=1, padding=3)
        if self.aux_data:
            self.sft_layer = SFTLayer(num_processing_features, condition_channels)
        
        self.res2_4 = nn.Sequential(*[ResidualBlock(num_processing_features, num_processing_features) for _ in range(3)])
        self.conv5 = nn.Conv2d(num_processing_features, num_processing_features*2, kernel_size=5, stride=2, padding=2)
        self.res6_8 = nn.Sequential(*[ResidualBlock(num_processing_features*2, num_processing_features*2) for _ in range(3)])
        self.conv9 = nn.Conv2d(num_processing_features*2, num_processing_features*4, kernel_size=3, stride=2, padding=1)
        self.res10_15 = nn.Sequential(*[ResidualBlock(num_processing_features*4, num_processing_features*4) for _ in range(6)])
        self.unconv16 = nn.ConvTranspose2d(num_processing_features*4, num_processing_features*2, kernel_size=4, stride=2, padding=1)
        self.res17_19 = nn.Sequential(*[ResidualBlock(num_processing_features*2, num_processing_features*2) for _ in range(3)])
        self.unconv20 = nn.ConvTranspose2d(num_processing_features*2, num_processing_features, kernel_size=4, stride=2, padding=1)
        self.res21_23 = nn.Sequential(*[ResidualBlock(num_processing_features, num_processing_features) for _ in range(3)])
        self.conv24 = nn.Conv2d(num_processing_features, 3*out_frames, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

        
    def forward(self, inputs):
        if self.aux_data:
            x, z = inputs
            z = (z - 0.5) * 2 # make sure the input is in range [-1, 1]
            assert x.shape[3:] == z.shape[3:], "x and z must have the same spatial dimensions"
            assert x.shape[1] == z.shape[1], "x and z must have the same number of consecutive frames"
        else:
            x = inputs[0]
        # make sure the input is in range [-1, 1]
        x = (x - 0.5) * 2
        # store the input for shortcut connection
        shortcut = x
        
        B, F, C, H, W = x.shape
        # reshape x to (B, F*C, H, W)
        x = x.reshape(B, F*C, H, W)
        # conv1
        x = self.conv1(x)

        # sft layer
        if self.aux_data:
            Cz = z.shape[2]
            if F > 1:
                stencil = (F - self.out_frames) // 2
                z = z[:,stencil:-stencil]
            z = z.reshape(B, self.out_frames*Cz, H, W)
            x = self.sft_layer(x, z)

        # res2-4
        res4 = self.res2_4(x)
        # conv5
        x = self.conv5(res4)
        # res6-8
        res8 = self.res6_8(x)
        # conv9
        x = self.conv9(res8)
        # res10-15
        res15 = self.res10_15(x)
        # unconv16
        x = self.unconv16(res15) + res8
        # res17-19
        x = self.res17_19(x)
        # unconv20
        x = self.unconv20(x) + res4
        # res21-23
        x = self.res21_23(x)
        # conv24
        x = self.conv24(x)
        x = x.reshape(B, self.out_frames, C, H, W)

        if F > 1:
            stencil = (F - self.out_frames) // 2
            shortcut = shortcut[:,stencil:-stencil]

        # Final output
        x = self.tanh(x) + shortcut


        # make sure the output is in range [0, 1]
        x = (x + 1) / 2
        
        return x