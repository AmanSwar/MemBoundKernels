#pragma once

#include <cuda_runtime.h>
#include <cuda_pipeline.h>

__global__ void async_embedding_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ embedding_matrix,
    float* __restrict__ output,
    int num_indices,
    int embedding_dim
) {
    int row_idx = blockIdx.x;
    if (row_idx >= num_indices) return;

    int embedding_idx = indices[row_idx];
    const float* src_row = embedding_matrix + (embedding_idx * embedding_dim);
    float* dst_row = output + (row_idx * embedding_dim);

    extern __shared__ float smem_buffer[];
    int vec_dim = embedding_dim / 4; 

    // async load Global -> Shared
    for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        __pipeline_memcpy_async(&smem_buffer[i * 4], &src_row[i * 4], 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // store shared -> Global
    for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        float4 tmp = reinterpret_cast<float4*>(&smem_buffer[i * 4])[0];
        reinterpret_cast<float4*>(&dst_row[i * 4])[0] = tmp;
    }
}
