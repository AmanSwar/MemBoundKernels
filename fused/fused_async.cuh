#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#define WARP_SIZE 32

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

__global__ void embed_rmsnorm_async_kernel(
    const int* __restrict__ indices,
    const half2* __restrict__ embedding_matrix,
    const half2* __restrict__ weight_ptr,
    half2* __restrict__ output,
    int num_indices,
    int embedding_dim,
    float eps = 1e-6f
) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    
    // 1. Setup pointers
    int index = indices[bx];
    int vcols = embedding_dim / 2; // number of half2 elements
    int embedStart = index * vcols;

    extern __shared__ char shared_mem[];
    half2* smem = reinterpret_cast<half2*>(shared_mem);
    // Ensure float pointer is aligned. vcols*4 is always div by 4, so we are safe.
    float* smem_partial_sum = reinterpret_cast<float*>(smem + vcols);

    // 2. Async Copy: Global -> Shared
    // We vectorize to int4 (16 bytes) to saturate the bus.
    // 1 int4 = 4 half2 elements.
    const int4* src_ptr_int4 = reinterpret_cast<const int4*>(embedding_matrix + embedStart);
    int4* dst_ptr_int4 = reinterpret_cast<int4*>(smem);
    
    int vcols_int4 = vcols / 4; 
    int remainder_half2 = vcols % 4; // Handle edges if dim not div by 8

    // Pipeline copies
    for (int idx = tid; idx < vcols_int4; idx += blockDim.x) {
        // cp.async.ca.shared.global [dst], [src], 16;
        // 'ca' = cache all (L2), usually best for one-time reads like embeddings
        size_t smem_addr = __cvta_generic_to_shared(&dst_ptr_int4[idx]);
        __pipeline_memcpy_async(reinterpret_cast<void*>(smem_addr), 
                                &src_ptr_int4[idx], 
                                16); 
    }

    // Handle Remaining half2s (if embedding_dim is not divisible by 8)
    if (remainder_half2 > 0) {
        int start_rem = vcols_int4 * 4;
        for (int idx = tid; idx < remainder_half2; idx += blockDim.x) {
             // Fallback to standard scalar load for tails
             // (cp.async supports smaller sizes, but scalar is fine for tails)
             smem[start_rem + idx] = embedding_matrix[embedStart + start_rem + idx];
        }
    }

    // 3. Commit and Wait
    __pipeline_commit();
    __pipeline_wait_prior(0); // Wait for ALL batches to finish
    __syncthreads(); // Ensure all threads see the data in smem

    // 4. Compute Sum of Squares (Read from Smem)
    float partial = 0.0f;
    for (int idx = tid; idx < vcols; idx += blockDim.x) {
        half2 element = smem[idx];
        float fx = __half2float(element.x);
        float fy = __half2float(element.y);
        partial += (fx * fx) + (fy * fy);
    }

    // 5. Reduction (Same as your code)
    float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);
    float inv_rms = rsqrtf((total_sum / float(embedding_dim)) + eps);

    // 6. Normalize & Write Back
    // We could vectorize writes too, but computations are usually the limit here.
    for (int idx = tid; idx < vcols; idx += blockDim.x) {
        half2 element = smem[idx];
        half2 w = weight_ptr[idx]; 

        float fx = __half2float(element.x) * inv_rms;
        float fy = __half2float(element.y) * inv_rms;

        float out_x = fx * __half2float(w.x);
        float out_y = fy * __half2float(w.y);

        output[bx * vcols + idx] = __floats2half2_rn(out_x, out_y);
    }
}