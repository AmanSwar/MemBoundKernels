#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>


#include "kernels/rmsnorm_naive.cuh"
#include "kernels/rmsnorm_vec.cuh"
#include "kernels/rmsnorm_optim.cuh"


void rmsnorm_cpu(float *input_matrix, float *weight_matrix,
                 float *output_matrix, int M, int N, float eps) {

  for (int i = 0; i < M; i++) {
    float sum = 0.0f;

    for (int j = 0; j < N; j++) {
      float elem = input_matrix[i * N + j];
      sum += (elem * elem);
    }

    float rms = std::sqrt((sum / N) + eps);

    for (int j = 0; j < N; j++) {
      output_matrix[i * N + j] =
          (input_matrix[i * N + j] / rms) * weight_matrix[j];
    }
  }
}

void verify(float *kernel_output, float *cpu_output, int M, int N,
            float tolerance = 1e-3) {
  for (int i = 0; i < M * N; i++) {
    if (std::abs(kernel_output[i] - cpu_output[i]) > tolerance) {
      std::cout << "FAIL" << std::endl;
      std::cout << "Error at index " << i
                << ": Kernel out: " << kernel_output[i]
                << " CPU out: " << cpu_output[i]
                << " Diff: " << std::abs(kernel_output[i] - cpu_output[i])
                << std::endl;
      return;
    }
  }

  std::cout << "PASS" << std::endl;
  return;
}

void init(float *matrix, int N) {
  for (int i = 0; i < N; i++) {
    matrix[i] = 1.0f + static_cast<float>(rand()) / RAND_MAX *
                           99.0f; // Random float between 1 and 100
  }
}

template <typename type>
void benchmark(void (*function)(const type *, const type *,
                           type *, int, int, float),
          std::string function_name, int M, int N, float eps) {

  float *input_mat = new float[M * N];
  float *out_mat = new float[M * N];
  float *weight = new float[N];

  srand(42);
  init(weight, N);
  init(input_mat, M * N);
  float *input_q = new float[M * N];
  float *weight_q = new float[N];

  for (int i = 0; i < M * N; ++i){
    input_q[i] = __half2float(__float2half(input_mat[i]));
  }
  for (int j = 0; j < N; ++j){
    weight_q[j] = __half2float(__float2half(weight[j]));
  }

  rmsnorm_cpu(input_q, weight_q, out_mat, M, N, eps);
  delete[] input_q;
  delete[] weight_q;

  // rmsnorm_cpu(input_mat, weight, out_mat, M, N , eps);

  type *da, *dw, *dout;
  cudaMalloc(&da, sizeof(type) * M * N);
  cudaMalloc(&dw, sizeof(type) * N);
  cudaMalloc(&dout, sizeof(type) * M * N);

  type *ha = new type[M * N];
  type *hw = new type[N];
  type *hout = new type[M * N];

  for (int i = 0; i < M * N; i++) {
    ha[i] = __float2half(input_mat[i]);
  }

  for (int i = 0; i < N; i++) {
    hw[i] = __float2half(weight[i]);
  }

  cudaMemcpy(da, ha, sizeof(type) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw, sizeof(type) * N, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaEvent_t start_event, end_event;

  cudaStreamCreate(&stream);
  cudaEventCreate(&start_event);
  cudaEventCreate(&end_event);

  function(da, dw, dout, M, N, eps);
  cudaDeviceSynchronize();

  const int num_runs = 100;

  cudaEventRecord(start_event, stream);

  for (int run = 0; run < num_runs; run++) {
    function(da, dw, dout, M, N, eps);
  }

  cudaEventRecord(end_event, stream);
  cudaEventSynchronize(end_event);

  float total_time_ms;
  cudaEventElapsedTime(&total_time_ms, start_event, end_event);
  float time_ms = total_time_ms / num_runs; // Average time per run in ms
  
  double total_bytes_transferred = (M * N * 2) * 2 + 2 * M;
  double gbs = (total_bytes_transferred * 1e-6)/ time_ms; 
  
  long long total_ops = 4LL * M * N;
  float gflops = (total_ops / (time_ms * 1e9)) * 1000.0f; // GFLOPS

  cudaMemcpy(hout, dout, sizeof(type) * M * N, cudaMemcpyDeviceToHost);

  float *kernel_output = new float[M * N];
  for (int i = 0; i < M * N; i++) {
    kernel_output[i] = __half2float(hout[i]);
  }

  std::cout << "Function: " << function_name << std::endl;
  std::cout << "Matrix size: " << M << "x" << N << std::endl;
  std::cout << "Time: " << time_ms << " ms" << std::endl;
  std::cout << "GFLOPS: " << gflops << std::endl;
  std::cout << "Gb/s : " << gbs << std::endl;

  std::cout << "Verification: ";
  verify(kernel_output, out_mat, M, N, 1e-1); // Relaxed tolerance for bfloat16

  delete[] input_mat;
  delete[] out_mat;
  delete[] weight;
  delete[] ha;
  delete[] hw;
  delete[] hout;
  delete[] kernel_output;

  cudaFree(da);
  cudaFree(dw);
  cudaFree(dout);

  cudaEventDestroy(start_event);
  cudaEventDestroy(end_event);
  cudaStreamDestroy(stream);

  std::cout << "----------------------------------------" << std::endl;
}

int main(){
  int N = 2048;
  double eps = 1e-6f;
  for(int i = 1 ; i <=10 ; i++){ 
    int M = 128 * 8 * i;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Dimension : " << M << "x " << N << std::endl;
    benchmark<half>(launch_rms_fp16 , "Naive" , M , N , eps);
    // benchmark<half>(launch_rmsnorm_fp16_vectorized , "Vectorized" , M , N , eps);
    benchmark<half>(launch_vec_rmsnorm , "Vectorized" , M , N , eps);
  }
}