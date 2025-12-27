#include <cuda_runtime.h>


// __launch_bounds__(256,4)
__global__ void float4_embedding_kernel(
  const int* __restrict__ indices,
  float4* __restrict__ embedding_matrix,
  float4* __restrict__ output,
  int num_indices,
  int embedding_dim
){
  int rowStart = blockIdx.x * embedding_dim;
  int index = indices[blockIdx.x];
  int embedStart = index * embedding_dim;
  #pragma unroll
  for(int d = threadIdx.x; d < embedding_dim; d += blockDim.x){
    output[rowStart + d] = embedding_matrix[embedStart + d];
  }

}


void launch_float4_embedding(
  const int* indices,
  float* embedding_matrix,
  float* output,
  int num_indices,
  int embedding_dim
){
  int threadsPerBlock = 1024;
  int blocksPerGrid = num_indices;

  float4* embedMatrixF4 = reinterpret_cast<float4* >(embedding_matrix);
  float4* outputF4 = reinterpret_cast<float4* >(output);

  float4_embedding_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    indices,
    embedMatrixF4,
    outputF4,
    num_indices,
    embedding_dim / 4
  );
}
