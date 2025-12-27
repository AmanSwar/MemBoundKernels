#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#define WARP_SIZE 32
#define LD32BITS(value) (reinterpret_cast<const half2 *>(&(value))[0])
#define ST32BITS(addr, value) (reinterpret_cast<half2 *>(&(addr))[0] = (value))

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  const unsigned int MASK = 0xffffffffu;
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset >= 1; offset >>= 1) {
    v += __shfl_xor_sync(MASK, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_f32(float *sdata, float val) {
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int wid = tid / WARP_SIZE; // warp id

  val = warp_reduce_sum_f32(val);

  if (lane == 0)
    sdata[wid] = val;

  __syncthreads();

  float total = 0.0f;
  if (wid == 0) {
    float v = (lane < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? sdata[lane]
                                                                  : 0.0f;
    v = warp_reduce_sum_f32(v);
    if (lane == 0)
      sdata[0] = v;
  }

  __syncthreads();
  total = sdata[0];
  return total;
}


__global__ void embed_rmsnorm_naive_kernel(
  const int* __restrict__ indices,
  const half2* __restrict__ embedding_matrix,
  const half2* __restrict__ weight_ptr,
  half2* __restrict__ output,
  int num_indices,
  int embedding_dim,
  float eps = 1e-6f
){

  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int index = indices[bx];

  int vcols = embedding_dim / 2;
  
  int embedStart = index * vcols;

  extern __shared__ char shared_mem[];
  half2* smem = reinterpret_cast<half2* >(shared_mem);
  float* smem_partial_sum = reinterpret_cast<float* >(smem+ vcols);

  float partial = 0.0f;
  for(int idx = tid ; idx < vcols ; idx += blockDim.x){
    half2 element = embedding_matrix[embedStart + idx];
    float fx = __half2float(element.x);
    float fy = __half2float(element.y);
    partial += (fx * fx) + (fy * fy);
    smem[idx] = element;
  }

  float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);

  float inv_rms = rsqrtf((total_sum / float(embedding_dim)) + eps);

  for (int idx = tid; idx < vcols; idx += blockDim.x) {
    half2 element = smem[idx];
    half2 w =
        weight_ptr[idx]; 

    float fx = __half2float(element.x) * inv_rms;
    float fy = __half2float(element.y) * inv_rms;

    float out_x = fx * __half2float(w.x);
    float out_y = fy * __half2float(w.y);

    half2 store;
    store.x = __float2half(out_x);
    store.y = __float2half(out_y);

    output[bx * vcols + idx] = store;
  }


}


void launch_embed_rmsnorm_fused(
  const int* indices,
  half* embedding_matrix,
  const half* weight_matrix,
  half* out_matrix,
  int num_indices , int embed_dim 
){
  int threads_per_block = 256;
  int blocks_per_grid = num_indices;

  int vcols = (embed_dim + 1) / 2;

  size_t cache_size = vcols * sizeof(half2);
  size_t reduction_size = WARP_SIZE * sizeof(float);
  size_t smem_size = cache_size + reduction_size;

  const half2* embedding_ptr = reinterpret_cast<const half2* >(embedding_matrix);
  const half2* weight_ptr = reinterpret_cast<const half2 * >(weight_matrix);
  half2* output_ptr = reinterpret_cast<half2* >(out_matrix);

  embed_rmsnorm_naive_kernel<<<blocks_per_grid , threads_per_block , smem_size>>>(
    indices,
    embedding_ptr,
    weight_ptr,
    output_ptr,
    num_indices , embed_dim
  );


}

