#include <cuda_runtime.h>
#include <iostream>


#include "kernels/better_embed.cuh"


int main(){

  //verify kernel output
  const int N = 128;
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
  cudaMalloc(&d_indices, N * sizeof(int));
  cudaMalloc(&d_embedding_matrix, VOCAB_SIZE * DIM * sizeof(float));
  cudaMalloc(&d_output, N * DIM * sizeof(float));

  cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_embedding_matrix, h_embedding_matrix, VOCAB_SIZE * DIM * sizeof(float), cudaMemcpyHostToDevice);

  launch_vec_embedding(
    d_indices,
    d_embedding_matrix,
    d_output,
    N,
    DIM
  );

  cudaMemcpy(h_output_gpu, d_output, N * DIM * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout <<"success\n";
}