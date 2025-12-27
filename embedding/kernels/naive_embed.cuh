#pragma once
#include <cuda_runtime.h>


// __launch_bounds__(256,4)
__global__ void naive_embedding_kernel(
  const int* __restrict__ indices,
  float* __restrict__ embedding_matrix,
  float* __restrict__ output,
  int num_indices,
  int embedding_dim
){


  int rowStart = blockIdx.x * embedding_dim;
  int index = indices[blockIdx.x];
  int embedStart = index * embedding_dim;

  // #pragma unroll
  for(int d = threadIdx.x; d < embedding_dim; d += blockDim.x){
    output[rowStart + d] = __ldg(&embedding_matrix[embedStart + d]);
  }

}


void launch_naive_embedding(
  const int* indices,
  float* embedding_matrix,
  float* output,
  int num_indices,
  int embedding_dim
){
  int threadsPerBlock = 1024;
  int blocksPerGrid = num_indices;

  naive_embedding_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    indices,
    embedding_matrix,
    output,
    num_indices,
    embedding_dim
  );
}
