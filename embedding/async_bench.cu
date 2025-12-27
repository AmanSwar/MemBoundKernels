
#include <cuda_runtime.h>
#include <cuda_pipeline.h> // For Async kernel
#include <iostream>
#include <vector>
#include <iomanip>


#include "kernels/naive_embed.cuh"
#include "kernels/vec_embed.cuh"
#include "kernels/fp32x4_unroll_embed.cuh"
#include "kernels/fp32_f4_pure_embed.cuh"
#include "kernels/sm_embed.cuh"
#include "kernels/async_embed.cuh"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ==========================================
// 1. NAIVE (Baseline)
// ==========================================
// __global__ void naive_embedding_kernel(
//     const int* __restrict__ indices,
//     const float* __restrict__ embedding_matrix,
//     float* __restrict__ output,
//     int num_indices,
//     int embedding_dim
// ){
//     int rowStart = blockIdx.x * embedding_dim;
//     int index = indices[blockIdx.x];
//     int embedStart = index * embedding_dim;
    
//     for(int d = threadIdx.x; d < embedding_dim; d += blockDim.x){
//         output[rowStart + d] = embedding_matrix[embedStart + d];
//     }
// }

// ==========================================
// 2. YOUR: Vectorized Embedding (2x Float4 Unroll)
// ==========================================
// __global__ void vec_embedding_kernel(
//   const int* __restrict__ indices,
//   float* __restrict__ embedding_matrix,
//   float* __restrict__ output,
//   int num_indices,
//   int embedding_dim
// ){
//   int rowStart = blockIdx.x * embedding_dim;
//   int index = indices[blockIdx.x];
//   int embedRowStart = index * embedding_dim;

//   const int threadPr = 8; // Processing 8 floats (2x float4) per iter
//   int limit = embedding_dim / threadPr;

//   #pragma unroll
//   for(int d = threadIdx.x; d < limit; d += blockDim.x){
//     // Access index logic fixed to match stride
//     int offset = d * threadPr; 
//     LDST128BITS(output[rowStart + offset]) = LDST128BITS(embedding_matrix[embedRowStart + offset]);
//     LDST128BITS(output[rowStart + offset + 4]) = LDST128BITS(embedding_matrix[embedRowStart + offset + 4]);
//   }

//   // Handle remainder (tail)
//   int processed = limit * threadPr;
//   for(int i = processed + threadIdx.x; i < embedding_dim; i += blockDim.x){
//       output[rowStart + i] = embedding_matrix[embedRowStart + i];
//   }
// }

// ==========================================
// 3. YOUR: Manual Loop Unroll (Scalar code)
// ==========================================
// __global__ void embedding_f32x4_kernel(const int* __restrict__ idx, float* __restrict__ weight,
//                              float* __restrict__ output, int n, int embed_dim) {
//   int thx = threadIdx.x;
//   int bx = blockIdx.x;
//   int index = idx[bx];
//   int rowStart = bx * embed_dim;
//   int embeddingStart = index * embed_dim;
  
//   for(int tx = thx * 4 ; tx < embed_dim ; tx += blockDim.x * 4){ 
//     if(tx + 3 < embed_dim) {
//         output[rowStart + tx] = weight[embeddingStart + tx];
//         output[rowStart + tx + 1] = weight[embeddingStart + tx + 1];
//         output[rowStart + tx + 2] = weight[embeddingStart + tx + 2];
//         output[rowStart + tx + 3] = weight[embeddingStart + tx + 3];
//     }
//     else{
//         for(int i = 0 ; tx + i < embed_dim ; i++){
//             output[rowStart + tx + i] = weight[embeddingStart + tx + i];
//         }
//     }
//   }
// }

// ==========================================
// 4. YOUR: Explicit Float4 Pointer Cast
// ==========================================
// __global__ void float4_embedding_kernel(
//   const int* __restrict__ indices,
//   float4* __restrict__ embedding_matrix,
//   float4* __restrict__ output,
//   int num_indices,
//   int embedding_dim
// ){
//   int rowStart = blockIdx.x * embedding_dim; // embedding_dim is now in float4 units
//   int index = indices[blockIdx.x];
//   int embedStart = index * embedding_dim;

//   for(int d = threadIdx.x; d < embedding_dim; d += blockDim.x){
//     output[rowStart + d] = embedding_matrix[embedStart + d];
//   }
// }

// ==========================================
// 5. YOUR: Shared Memory Staging (Sync)
// ==========================================
// __global__ void sm_embed_kernel(
//   const int* __restrict__ indices,
//   float* __restrict__ embed_matrix,
//   float* __restrict__ output,
//   int N, int embed_dim
// ){
//   extern __shared__ float4 smem[];
  
//   int rowStart = blockIdx.x * embed_dim;
//   int index = indices[blockIdx.x];

//   const int threadPr = 4;
//   const int limit = embed_dim / threadPr;
  
//   // Global -> Shared
//   for(int d = threadIdx.x ; d < limit ; d += blockDim.x){
//     smem[d] = LDST128BITS(embed_matrix[index * embed_dim + threadPr*d]);
//   }

//   __syncthreads();

//   // Shared -> Global
//   for(int d = threadIdx.x ; d < limit ; d += blockDim.x){
//      LDST128BITS(output[rowStart + d*threadPr]) = smem[d];
//   }
// }

// ==========================================
// 6. MY: Async Copy (Ampere+ Only)
// // ==========================================
// __global__ void async_embedding_kernel(
//     const int* __restrict__ indices,
//     const float* __restrict__ embedding_matrix,
//     float* __restrict__ output,
//     int num_indices,
//     int embedding_dim
// ) {
//     int row_idx = blockIdx.x;
//     if (row_idx >= num_indices) return;

//     int embedding_idx = indices[row_idx];
//     const float* src_row = embedding_matrix + (embedding_idx * embedding_dim);
//     float* dst_row = output + (row_idx * embedding_dim);

//     extern __shared__ float smem_buffer[];
//     int vec_dim = embedding_dim / 4; 

//     // Async Load Global -> Shared
//     for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
//         __pipeline_memcpy_async(&smem_buffer[i * 4], &src_row[i * 4], 16);
//     }
//     __pipeline_commit();
//     __pipeline_wait_prior(0);
//     __syncthreads();

//     // Store Shared -> Global
//     for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
//         float4 tmp = reinterpret_cast<float4*>(&smem_buffer[i * 4])[0];
//         reinterpret_cast<float4*>(&dst_row[i * 4])[0] = tmp;
//     }
// }


template <typename Func>
void run_benchmark(const char* name, Func kernel_launch, int num_indices, int embedding_dim, int repeats) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    kernel_launch();
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        kernel_launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / repeats;
    
    // Calculate Bandwidth (Read + Write)
    double total_bytes = (double)num_indices * sizeof(int) + (double)num_indices * embedding_dim * sizeof(float) * 2;
    double throughput_gbs = (total_bytes * 1e-9) / (avg_time_ms * 1e-3);

    std::cout << std::left << std::setw(30) << name 
              << " | Time: " << std::fixed << std::setprecision(5) << avg_time_ms << " ms"
              << " | BW: " << std::setprecision(2) << throughput_gbs << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int vocab_size = 151936;
    const int embedding_dim = 4096;

    float* d_table;

    size_t table_size = (size_t)vocab_size * embedding_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_table, table_size));

    std::vector<float> h_table(vocab_size * embedding_dim);
    for(size_t i = 0; i < h_table.size(); i++){
        h_table[i] = (float)(i % 100) / 100.0f;  
    }

    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), table_size, cudaMemcpyHostToDevice));

    std::cout << "--- Embedding Kernel Benchmark ---" << std::endl;
    for(int n = 1 ; n <= 10 ; n++){
        const int num_indices = 128 * 8*n;
        
        std::cout << "Dim: " << embedding_dim << " | Indices: " << num_indices << std::endl;
   
        size_t output_size = (size_t)num_indices * embedding_dim * sizeof(float);
        size_t indices_size = num_indices * sizeof(int);

        float* d_output;
        int * d_indices;

        CUDA_CHECK(cudaMalloc(&d_output, output_size));
        CUDA_CHECK(cudaMalloc(&d_indices, indices_size));


        std::vector<int> h_indices(num_indices);
        for(int i=0; i<num_indices; i++) h_indices[i] = rand() % vocab_size;

        CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), indices_size, cudaMemcpyHostToDevice));


        int blocks = num_indices;
        int threads = 256; // not for all

        //naive
        // run_benchmark("1. Naive/ Coalesced access", [&]() {
        //     naive_embedding_kernel<<<blocks, 1024>>>(d_indices, d_table, d_output, num_indices, embedding_dim);
        // }, num_indices, embedding_dim, 500);

        // // // Vec Embedding (2x Float4)
        // run_benchmark("2. Vec Embed (2x Float4)", [&]() {
        //     vec_embedding_kernel<<<blocks, 1024>>>(d_indices, d_table, d_output, num_indices, embedding_dim);
        // }, num_indices, embedding_dim, 500);

        // // // Manual Loop Unroll (Scalar)
        // run_benchmark("3. Manual Unroll (Scalar)", [&]() {
        //     embedding_f32x4_kernel<<<blocks, 1024>>>(d_indices, d_table, d_output, num_indices, embedding_dim);
        // }, num_indices, embedding_dim, 500);

        // // // 4. Float4 Pointer Cast
        // run_benchmark("4. Float4 Cast (Pure)", [&]() {
        //     // embedding_dim must be divided by 4 for this kernel
        //     float4_embedding_kernel<<<blocks, 1024>>>(d_indices, (float4*)d_table, (float4*)d_output, num_indices, embedding_dim/4);
        // }, num_indices, embedding_dim, 500);

        // // // 5. Shared Memory Staging
        // run_benchmark("5. SMEM Staged (Sync)", [&]() {
        //     size_t smem = embedding_dim * sizeof(float);
        //     sm_embed_kernel<<<blocks, 1024, smem>>>(d_indices, d_table, d_output, num_indices, embedding_dim);
        // }, num_indices, embedding_dim, 500);

        // // // 6. Async (Ampere Only)
        run_benchmark("6. Async Copy (Ampere)", [&]() {
            size_t smem = embedding_dim * sizeof(float);
            async_embedding_kernel<<<blocks, 1024, smem>>>(d_indices, d_table, d_output, num_indices, embedding_dim);
        }, num_indices, embedding_dim, 500);

        cudaFree(d_output);
        cudaFree(d_indices);
        printf("\n");
    }
    return 0;

    cudaFree(d_table);
}