#include <cuda_runtime.h>

__global__ void embedding_f32x4_kernel(const int* __restrict__ idx, float* __restrict__ weight,
                             float* __restrict__ output, int n, int embed_dim) {
  int thx = threadIdx.x;
  int bx = blockIdx.x;
  int index = idx[bx];
  int rowStart = bx * embed_dim;
  int embeddingStart = index * embed_dim;
  
  for(int tx = thx * 4 ; tx < embed_dim ; tx += blockDim.x * 4){ 
    if(tx + 3 < embed_dim) {
        output[rowStart + tx] = weight[embeddingStart + tx];
        output[rowStart + tx + 1] = weight[embeddingStart + tx + 1];
        output[rowStart + tx + 2] = weight[embeddingStart + tx + 2];
        output[rowStart + tx + 3] = weight[embeddingStart + tx + 3];
    }
    else{
        for(int i = 0 ; tx + i < embed_dim ; i++){
            output[rowStart + tx + i] = weight[embeddingStart + tx + i];
        }
    }
  }
}


void launch_fp32_vec_embed(
    const int *idx,
    float* weight,
    float* output,
    int N,
    int embed_size
){

    int blockDim {1024};
    int grid {N};
    embedding_f32x4_kernel<<<grid , blockDim>>>(idx, weight, output, N, embed_size);

}