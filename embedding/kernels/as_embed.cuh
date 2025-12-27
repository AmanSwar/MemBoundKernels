#include <cuda_runtime.h>
#include <cuda_pipeline.h> // Required for standard pipeline primitives

#define BLOCK_SIZE 256

// Helper for vectorized loads/stores
// We assume embedding_dim is divisible by 4 for float4. 
// If not, you need a scalar tail loop (omitted here for clarity).
__device__ __forceinline__ void copy_float4(float* dst, const float* src) {
    *((float4*)dst) = *((float4*)src);
}

__global__ void async_embedding_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ embedding_matrix,
    float* __restrict__ output,
    int num_indices,
    int embedding_dim
) {
    // 1. Setup pointers
    int row_idx = blockIdx.x; // One block per embedding row
    if (row_idx >= num_indices) return;

    int embedding_idx = indices[row_idx];
    
    // Offset calculation
    const float* src_row = embedding_matrix + (embedding_idx * embedding_dim);
    float* dst_row = output + (row_idx * embedding_dim);

    // 2. Allocate Shared Memory (Dynamic or Static)
    // Enough to hold one row. 
    // Example: 4096 dim * 4 bytes = 16KB (fits easily in smem)
    extern __shared__ float smem_buffer[];

    // 3. Pipeline Setup
    // Calculate how many float4 elements we have per row
    int vec_dim = embedding_dim / 4; 
    
    // Create a pipeline object (cuda::pipeline is cleaner, but using raw builtins for transparency)
    // We use a simple loop. No need for multi-stage pipeline since we just read -> write.

    // --------------------------------------------------------
    // STAGE 1: Global -> Shared (Async Copy)
    // --------------------------------------------------------
    for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        // Calculate offsets in bytes for cp.async
        size_t offset = i * 4 * sizeof(float);
        
        // __pipeline_memcpy_async(dst_shared, src_global, size_z)
        // src_row is global, smem_buffer is shared.
        // We load 16 bytes (float4) at a time.
        __pipeline_memcpy_async(&smem_buffer[i * 4], &src_row[i * 4], 16);
    }

    // Commit the copy commands
    __pipeline_commit();

    // --------------------------------------------------------
    // STAGE 2: Wait for Data
    // --------------------------------------------------------
    // Wait for all batches to finish (0 remaining)
    __pipeline_wait_prior(0);

    // Sync threads to ensure all data is visible in SMEM
    __syncthreads();

    // --------------------------------------------------------
    // STAGE 3: Shared -> Global (Write back)
    // --------------------------------------------------------
    // Note: There is no "async store" to global generally available.
    // We must load to registers, then store to global.
    
    for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        // Reinterpret cast to float4 for vectorized store
        float4 tmp = reinterpret_cast<float4*>(&smem_buffer[i * 4])[0];
        reinterpret_cast<float4*>(&dst_row[i * 4])[0] = tmp;
    }
}

void launch_as_embedding(
    const int* indices,
    float* embedding_matrix,
    float* output,
    int num_indices,
    int embedding_dim,
    cudaStream_t stream
) {
    // Shared memory size needed: embedding_dim * sizeof(float)
    size_t smem_size = embedding_dim * sizeof(float);

    async_embedding_kernel<<<num_indices, BLOCK_SIZE, smem_size, stream>>>(
        indices,
        embedding_matrix,
        output,
        num_indices,
        embedding_dim
    );
}