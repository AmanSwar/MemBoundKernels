#include <cuda_runtime.h>


#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void vec_embedding_kernel(
  const int* __restrict__ indices,
  float* __restrict__ embedding_matrix,
  float* __restrict__ output,
  int num_indices,
  int embedding_dim
){
  int rowStart = blockIdx.x * embedding_dim;
  int index = indices[blockIdx.x];
  int embedRowStart = index * embedding_dim;
  const int threadPr = 8;
  int limit = embedding_dim / threadPr;

  #pragma unroll
  for(int d = threadIdx.x; d < limit; d += blockDim.x){
    LDST128BITS(output[rowStart + d*threadPr])= LDST128BITS(embedding_matrix[embedRowStart + d*threadPr]);
    LDST128BITS(output[rowStart + d*threadPr + 4])= LDST128BITS(embedding_matrix[embedRowStart + d*threadPr + 4]);    
  }

  int processed = limit * threadPr;
  for(int i = processed + threadIdx.x; i < embedding_dim; i += blockDim.x){
      output[rowStart + i] = embedding_matrix[embedRowStart + i];
  }
}


void launch_vec_embedding(
  const int* indices,
  float* embedding_matrix,
  float* output,
  int num_indices,
  int embedding_dim
){
  int threadsPerBlock = 1024;
  int blocksPerGrid = num_indices;

  vec_embedding_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    indices,
    embedding_matrix,
    output,
    num_indices,
    embedding_dim
  );
  // cudaDeviceSynchronize();
}