#include <cstddef>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include "utils.cuh"

#include "kernels/naive_embed.cuh"
#include "kernels/vec_fp32_embed.cuh"
#include "kernels/better_embed.cuh"
#include "kernels/sm_embed.cuh"
#include "kernels/vec_embed.cuh"

#include "kernels/async_embed.cuh"
#include "kernels/as_embed.cuh"

void cpu_embedding(
  const int* indices,
  const float* embedding_matrix,
  float* output,
  int num_indices,
  int embedding_dim
){
  for(int i = 0; i < num_indices; i++){
    int index = indices[i];
    for(int d = 0; d < embedding_dim; d++){
      output[i * embedding_dim + d] = embedding_matrix[index * embedding_dim + d];
    }
  }
}


void benchmarkKernel(void(*function)(
  const int*,
  float*,
  float*,
  int,
  int
),
const char* name,
int* indices,
float* embedding_matrix,
float* output,
int num_indices,
int embedding_dim,
float* dummy , int dummySize
){
  const int iter = 100;

  std::cout << "Kernel : " << name << std::endl;
  //warmup
  for(size_t i{0} ; i < 5 ; ++i){
    function(
      indices,
      embedding_matrix,
      output,
      num_indices,
      embedding_dim
    ); 
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float totalMs = 0;
  for(size_t i{0} ; i < iter ; ++i){
    flush_l2(dummy , dummySize);
    
    cudaEventRecord(start);
    function(
      indices,
      embedding_matrix,
      output,
      num_indices,
      embedding_dim
    );
    
  cudaEventRecord(stop);
 
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  totalMs += milliseconds;
  }

  float avgMs = totalMs/ iter;
  double bytes = (double)num_indices * sizeof(int) + (double)num_indices * embedding_dim * sizeof(float) * 2;
  double gb_s = (bytes / 1e9) / (avgMs / 1e3);

  printf("Avg Execution Time (Cold L2): %f ms\n", avgMs);
  printf("Effective Throughput: %f GB/s\n\n", gb_s);  
  
  printf("\n");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void verify(float* kernelOut , float* cpuOut , int N){
  float max_diff = 0.0f;
  for(int i = 0; i < N; i++){
    float diff = fabs(kernelOut[i] - cpuOut[i]);
    if(diff > max_diff){
      max_diff = diff;
    }
  }
  printf("Max difference between GPU and CPU: %f\n", max_diff); 
}



int main(){

  //verify kernel output
  const int N = 1024;
  const int DIM = 4096;
  const int VOCAB_SIZE = 151936;

  int* h_indices = new int[N];
  float* h_embedding_matrix = new float[VOCAB_SIZE * DIM];
  float* h_output_gpu = new float[N * DIM];
  float* h_output_cpu = new float[N * DIM];

  for(int i = 0; i < N; i++){
    h_indices[i] = rand() % VOCAB_SIZE;
  }
  for(int i = 0; i < VOCAB_SIZE * DIM; i++){
    h_embedding_matrix[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }
  int* d_indices;
  float* d_embedding_matrix;
  float* d_output;
  float* d_dummy;
  int dummySize = 20 * 1024 * 1024 / sizeof(float);

  cudaMalloc(&d_indices, N * sizeof(int));
  cudaMalloc(&d_embedding_matrix, VOCAB_SIZE * DIM * sizeof(float));
  cudaMalloc(&d_output, N * DIM * sizeof(float));
  cudaMalloc(&d_dummy , dummySize * sizeof(float));

  cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_embedding_matrix, h_embedding_matrix, VOCAB_SIZE * DIM * sizeof(float), cudaMemcpyHostToDevice);

  
  // launch_naive_embedding(
  //   d_indices,
  //   d_embedding_matrix,
  //   d_output,
  //   N,
  //   DIM
  // );
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));


  // cudaMemcpy(h_output_gpu, d_output, N * DIM * sizeof(float), cudaMemcpyDeviceToHost);

  // cpu_embedding(
  //   h_indices,
  //   h_embedding_matrix,
  //   h_output_cpu,
  //   N,
  //   DIM
  // );
  // //verify correctness
  // verify(h_output_gpu, h_output_cpu, N * DIM);
  //benchmark kernel
  benchmarkKernel(launch_naive_embedding , "Naive",d_indices , d_embedding_matrix , d_output , N , DIM,d_dummy , dummySize);
  // benchmarkKernel(launch_fp32_vec_embed , "vec fp32 embedding" ,d_indices , d_embedding_matrix , d_output , N , DIM,d_dummy , dummySize);
  // benchmarkKernel(launch_vec_embedding , "double vec fp32 embedding" ,d_indices , d_embedding_matrix , d_output , N , DIM ,d_dummy , dummySize);
  // benchmarkKernel(launch_sm_embed_kernel , "SMEM embedding" ,d_indices , d_embedding_matrix , d_output , N , DIM , d_dummy , dummySize);
  // benchmarkKernel(launch_float4_embedding , "float4 embedding" ,d_indices , d_embedding_matrix , d_output , N , DIM , d_dummy , dummySize);
  // benchmarkKernel(launch_async_embedding , "async embedding" ,d_indices , d_embedding_matrix , d_output , N , DIM , d_dummy , dummySize);
  // benchmarkKernel(launch_as_embedding , "async embedding" ,d_indices , d_embedding_matrix , d_output , N , DIM , d_dummy , dummySize);


}