#include <cuda_runtime.h>

__global__ void flush_l2_kernel(float* dummy, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        // Volatile ensures the compiler doesn't optimize away the read
        volatile float val = dummy[id];
    }
}

void flush_l2(float* d_dummy, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    flush_l2_kernel<<<blocks, threads>>>(d_dummy, size);
    cudaDeviceSynchronize();
}