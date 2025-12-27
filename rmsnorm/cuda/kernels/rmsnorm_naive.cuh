#pragma once


#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "common.cuh"

__global__
void rms_norm_kernel_fp16(
    const half* __restrict__ input_matrix_ptr,
    const half* __restrict__ weight_ptr,
    half* __restrict__ out_matrix_ptr,
    int M, int N,
    float eps
){
  int row_index = blockIdx.x;
  if (row_index >= M) return;
  int row_start = row_index * N;

  extern __shared__ float sdata[];
  float partial = 0.0f;
  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
    float in_f = __half2float(input_matrix_ptr[row_start + idx]);
    partial += in_f * in_f;
  }

  float total_sum = block_reduce_sum_f32(sdata, partial);

  float rms = sqrtf((total_sum / N) + eps);

  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
    float in_f = __half2float(input_matrix_ptr[row_start + idx]);
    float w_f = __half2float(weight_ptr[idx]); 
    float out_f = (in_f / rms) * w_f;
    out_matrix_ptr[row_start + idx] = __float2half(out_f);
  }
}

void launch_rms_fp16(
    const half *input_matrix,const half *weight_matrix,
    half *out_matrix, int M, int N,
    float eps = 1e-6f
){
  int threads_per_block = 256; 
  int blocks_per_grid = M;
  int NUM_WARPS = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;

  size_t smem_size = (threads_per_block + NUM_WARPS) * sizeof(float);
  rms_norm_kernel_fp16<<<blocks_per_grid, threads_per_block, smem_size>>>(
        input_matrix, weight_matrix, out_matrix, M, N, eps
    );

}