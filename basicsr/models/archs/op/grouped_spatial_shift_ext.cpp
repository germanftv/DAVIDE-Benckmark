#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)


// Declaration from the CUDA kernel .cu file
torch::Tensor grouped_spatial_shift_cuda(const torch::Tensor& input, const torch::Tensor& shifts_x_tensor, const torch::Tensor& shifts_y_tensor);

// Wrapper function to be called from Python
torch::Tensor grouped_spatial_shift(const torch::Tensor& input, const torch::Tensor& shifts_x, const torch::Tensor& shifts_y) {
    // Check tensor requirements
    CHECK_INPUT(input);
    CHECK_INPUT(shifts_x);
    CHECK_INPUT(shifts_y);
    
    return grouped_spatial_shift_cuda(input, shifts_x, shifts_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_spatial_shift", &grouped_spatial_shift, "Multi-channel shift operation");
}
