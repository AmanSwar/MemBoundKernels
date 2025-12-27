#pragma once

#include <cstddef>
#include <cuda_runtime.h>


#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sm_embed_kernel(
  const int* __restrict__ indices,
  float* __restrict__ embed_matrix,
  float* __restrict__ output,
  int N, int embed_dim
){
  extern __shared__ float4 smem[];
  int rowStart = blockIdx.x * embed_dim;
  int index = indices[blockIdx.x];
  const int threadPr = 4;
  const int limit = embed_dim / threadPr;
  
  #pragma unroll
  for(int d = threadIdx.x ; d < limit ; d += blockDim.x){
    smem[d] = LDST128BITS(embed_matrix[index * embed_dim + threadPr*d]);
  }
  __syncthreads();
  #pragma unroll
  for(int d = threadIdx.x ; d < limit ; d += blockDim.x){
     LDST128BITS(output[rowStart + d*threadPr]) = smem[d];
  }
}

void launch_sm_embed_kernel(
  const int* indices,
  float* embedding_matrix,
  float* output,
  int num_indices,
  int embedding_dim
){

  if(embedding_dim % 4 !=0) throw "Embed dim not divisible by 4";
  int threadPerBlock = 1024;
  int blockPerGrid = num_indices;
  size_t smem_size = embedding_dim * sizeof(float);
  sm_embed_kernel<<<blockPerGrid , threadPerBlock , smem_size>>>(
    indices , embedding_matrix , output , num_indices , embedding_dim
  );


}