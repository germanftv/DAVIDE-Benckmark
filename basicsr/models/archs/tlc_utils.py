# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is modified from:
# - TLC (https://github.com/megvii-research/TLC)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
# This code implements the utility functions for the Test-time Local Converter (TLC) strategy
# to enhance the performance of image restoration models during inference.
#
# Reference:
#     Chu, Xiaojie, et al. "Improving image restoration by revisiting global information aggregation."
#     European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.models.archs.arch_utils import CrossAttentionFusionBlock


class AvgPool2d(nn.Module):
    """Local AvgPool2d for TLC."""
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


class LocalInstanceNorm2d(nn.Module):
    """Local Instance Normalization for TLC."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        super().__init__()
        assert not track_running_stats
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.avgpool = AvgPool2d()
        self.eps = eps

    def forward(self, input):
        mean_x = self.avgpool(input) # E(x)
        mean_xx = self.avgpool(torch.mul(input, input)) # E(x^2)
        mean_x2 = torch.mul(mean_x, mean_x) # (E(x))^2
        var_x = mean_xx - mean_x2 # Var(x) = E(x^2) - (E(x))^2
        mean = mean_x
        var = var_x
        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight.view(1,-1, 1, 1) + self.bias.view(1,-1, 1, 1)
        return input


class LocalCrossAttentionFusionBlock(CrossAttentionFusionBlock):
    """Local Cross Attention Fusion Block for TLC."""
    def __init__(self, n_feat_x, n_feat_z, num_heads, bias, base_size=None, kernel_size=None, fast_imp=False, train_size=None):
        super().__init__(n_feat_x, n_feat_z, num_heads, bias)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp
        self.train_size = train_size

    def grids(self, x, z):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        x_parts, z_parts = [], []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                x_parts.append(x[:, :, i:i + k1, j:j + k2])
                z_parts.append(z[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        x_parts = torch.stack(x_parts, dim=0)
        z_parts = torch.stack(z_parts, dim=0)
        self.idxes = idxes
        return x_parts, z_parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[:, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :, :]
            count_mt[:, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def _forward(self, q, kv):

        k,v = kv.chunk(2, dim=2)
        q = rearrange(q, 'p b (head c) h w -> p b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'p b (head c) h w -> p b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'p b (head c) h w -> p b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature[None, None, ...]
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out

    def _pad(self, x):
        b,c,h,w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1- h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (mod_pad_w//2, mod_pad_w-mod_pad_w//2, mod_pad_h//2, mod_pad_h-mod_pad_h//2)
        x = F.pad(x, pad, 'reflect')
        return x, pad

    def forward(self, x, z):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        b,c,h,w = x.shape

        kv = self.kv_dwconv(z)
        q = self.q_dwconv(x)
        
        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            q, kv = self.grids(q, kv) # convert to local windows 
            out = self._forward(q, kv)
            out = rearrange(out, 'p b head c (h w) -> p b (head c) h w', head=self.num_heads, h=q.shape[-2], w=q.shape[-1])
            out = self.grids_inverse(out) # reverse

        out = self.project_out(out)
        return out


def replace_layers(model, base_size, train_size, fast_imp, tlc_layers, **kwargs):
    """Replace layers in the model with local layers."""
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, tlc_layers=tlc_layers, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d) and 'local_avgpool' in tlc_layers:
            # replace with local avgpool
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

        if isinstance(m, CrossAttentionFusionBlock) and 'local_crossattn' in tlc_layers:
            # replace with local cross attention fusion block
            attn = LocalCrossAttentionFusionBlock(m.n_feat_x, m.n_feat_z, num_heads=m.num_heads, bias=m.bias, base_size=base_size, fast_imp=False, train_size=train_size)
            setattr(model, n, attn)
    
    # Make sure the model is in the same device
    model.to(next(model.parameters()).device)


class Local_Base():
    """Base class for local models."""
    def convert(self, *args, inputs, train_size, tlc_layers, **kwargs):
        replace_layers(self, *args, train_size=train_size, tlc_layers=tlc_layers, **kwargs)

        with torch.no_grad():
            self.forward(inputs)