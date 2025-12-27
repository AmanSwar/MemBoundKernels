

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.cuh"

#define LD32BITS(value) (reinterpret_cast<const half2 *>(&(value))[0])
#define ST32BITS(addr, value) (reinterpret_cast<half2 *>(&(addr))[0] = (value))

__global__ void __launch_bounds__(256,2)
rmsnorm_vec_kernel(
  const half* __restrict__ input_matrix,
  const half* __restrict__ weight,
  half* __restrict__ output_matrix,
  int M,
  int N,
  float eps
){
  int tid = threadIdx.x;
  int row = blockIdx.x;
  if (row >= M) return;

  int vcols = (N + 1) / 2;  
  int row_start = row * N;
  
  extern __shared__ char shared_mem[];
  half2* smem = reinterpret_cast<half2*>(shared_mem);
  float* smem_partial_sum = reinterpret_cast<float*>(smem + vcols);
  
  float partial = 0.0f;
  int full_quads = N / 4;
  bool has_tail = (N % 4) != 0;
  

  for(int idx = tid; idx < full_quads; idx += blockDim.x){
    half2 element1 = LD32BITS(input_matrix[row_start + idx * 4]);
    float fx1 = __half2float(element1.x);
    float fy1 = __half2float(element1.y);
    
    half2 element2 = LD32BITS(input_matrix[row_start + idx * 4 + 2]);
    float fx2 = __half2float(element2.x);
    float fy2 = __half2float(element2.y);
    
    partial += fx1 * fx1 + fy1 * fy1 + fx2 * fx2 + fy2 * fy2;
    
    smem[idx * 2] = element1;
    smem[idx * 2 + 1] = element2;
  }

  if(has_tail && tid == 0){

    int tail_idx = full_quads * 2; 
    int tail_start = full_quads * 4;
    int tail_count = N % 4;
    
    if(tail_count == 1){
      half tail_element = input_matrix[row_start + tail_start];
      float fx = __half2float(tail_element);
      partial += fx * fx;
      
      half2 new_el;
      new_el.x = tail_element;
      new_el.y = __float2half(0.0f);
      smem[tail_idx] = new_el;
      
    } else if(tail_count == 2){
      half2 tail_element = LD32BITS(input_matrix[row_start + tail_start]);
      float fx = __half2float(tail_element.x);
      float fy = __half2float(tail_element.y);
      partial += fx * fx + fy * fy;
      
      smem[tail_idx] = tail_element;
      
    } else if(tail_count == 3){

      half2 tail_element = LD32BITS(input_matrix[row_start + tail_start]);
      float fx1 = __half2float(tail_element.x);
      float fy1 = __half2float(tail_element.y);
      

      half tail_element2 = input_matrix[row_start + tail_start + 2];
      float fx2 = __half2float(tail_element2);
      
      partial += fx1 * fx1 + fy1 * fy1 + fx2 * fx2;
      

      smem[tail_idx] = tail_element;
      half2 last_el;
      last_el.x = tail_element2;
      last_el.y = __float2half(0.0f);
      smem[tail_idx + 1] = last_el;
    }
  }
  
  __syncthreads();
  

  float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);
  
  float inv_rms = rsqrtf((total_sum / float(N)) + eps);
  
  for(int idx = tid; idx < vcols; idx += blockDim.x){
    half2 element = smem[idx];
    half2 w = LD32BITS(weight[idx * 2]);
    
    float fx = __half2float(element.x) * inv_rms;
    float fy = __half2float(element.y) * inv_rms;
    
    float out_x = fx * __half2float(w.x);
    float out_y = fy * __half2float(w.y);
    
    half2 out_el;
    out_el.x = __float2half(out_x);
    out_el.y = __float2half(out_y);
    
    ST32BITS(output_matrix[row_start + idx * 2], out_el);
  }
}

void launch_vec_rmsnorm(
    const half* input_matrix,
    const half* weight,
    half* output_matrix,
    int M,
    int N,
    float eps
){
    int vcols = (N + 1) / 2;  
    size_t shared_mem_size = vcols * sizeof(half2) + 256 * sizeof(float);
    
    dim3 grid(M);
    dim3 block(256);
    
    rmsnorm_vec_kernel<<<grid, block, shared_mem_size>>>(
        input_matrix,
        weight,
        output_matrix,
        M,
        N,
        eps
    );
}