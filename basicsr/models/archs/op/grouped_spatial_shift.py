# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
gss = load(
    name='grouped_spatial_shift',
    sources=[
        os.path.join(module_path, 'grouped_spatial_shift_ext.cpp'),
        os.path.join(module_path, 'grouped_spatial_shift_kernel.cu'),
    ],
)


class GroupedSpatialShiftFunction(Function):
    @staticmethod
    def forward(ctx, input, shifts_x_tensor, shifts_y_tensor):
        # Store for backward
        input = input.contiguous()
        ctx.save_for_backward(shifts_x_tensor, shifts_y_tensor)
        ctx.input_shape = input.shape
        
        return gss.grouped_spatial_shift(input, shifts_x_tensor, shifts_y_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        shifts_x_tensor, shifts_y_tensor = ctx.saved_tensors
        batch_size, channels, height, width = ctx.input_shape

        # Compute inverse shifts for backward pass
        inverse_shifts_x = -shifts_x_tensor
        inverse_shifts_y = -shifts_y_tensor

        # Ensure the grad_output tensor is contiguous
        grad_output = grad_output.contiguous()

        # Perform the shift operation in reverse using the inverse shifts
        grad_input = gss.grouped_spatial_shift(grad_output, inverse_shifts_x, inverse_shifts_y)
        
        # No gradients for shifts since they're constant
        return grad_input, None, None


# To use this in your model
class GroupedSpatialShift(torch.nn.Module):
    def __init__(self, shifts_x_tensor, shifts_y_tensor):
        super(GroupedSpatialShift, self).__init__()
        # register shifts as buffers
        self.register_buffer('shifts_x_tensor', shifts_x_tensor)
        self.register_buffer('shifts_y_tensor', shifts_y_tensor)

    def forward(self, x):
        return GroupedSpatialShiftFunction.apply(x, self.shifts_x_tensor, self.shifts_y_tensor)

