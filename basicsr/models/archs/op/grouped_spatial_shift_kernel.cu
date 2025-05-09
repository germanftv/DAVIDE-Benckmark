// grouped_spatial_shift_kernel.cu
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void grouped_spatial_shift_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input,
                                           const int width, const int height, const int channels, const int batch_size,
                                           const int* __restrict__ shifts_x, const int* __restrict__ shifts_y) {
    int batchChan = blockIdx.x * blockDim.x + threadIdx.x; // Combined batch and channel index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.z + threadIdx.z;


    if (row < height && col < width && batchChan < batch_size * channels) {
        int batch = batchChan / channels; // Compute batch index
        int chan = batchChan % channels; // Compute channel index within batch

        int shift_x = shifts_x[chan];
        int shift_y = shifts_y[chan];
        
        int new_col = col + shift_y;
        int new_row = row + shift_x;

        size_t idx_in = (batch * channels + chan) * height * width + row * width + col;
        size_t idx_out = (batch * channels + chan) * height * width + new_row * width + new_col;

        if (new_col >= 0 && new_col < width && new_row >= 0 && new_row < height) {
            output[idx_out] = input[idx_in];
        }
    }
}

torch::Tensor grouped_spatial_shift_cuda(const torch::Tensor& input, const torch::Tensor& shifts_x_tensor, const torch::Tensor& shifts_y_tensor) {
    auto x = input.contiguous();
    const int* shifts_x = shifts_x_tensor.data_ptr<int>();
    const int* shifts_y = shifts_y_tensor.data_ptr<int>();

    const auto batch_size = x.size(0);
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    auto y = torch::zeros_like(x);

    const dim3 threadsPerBlock(16, 8, 8); // Example configuration, adjust based on your GPU's capability
    const dim3 numBlocks((batch_size * channels + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                         (width + threadsPerBlock.z - 1) / threadsPerBlock.z);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "grouped_spatial_shift_kernel", [&] {
        grouped_spatial_shift_kernel<scalar_t><<<numBlocks, threadsPerBlock, 0, stream>>>(
            y.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            width, height, channels, batch_size,
            shifts_x, shifts_y
        );
    });

    return y;
}
