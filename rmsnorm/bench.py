import rmsnorm_kernel
from layernorm.triton.rmsnorm import RMSNormTriton

def benchmark_rmsnorm(input_tensor, weight, eps=1e-6):
    import time

    # Warm-up
    for _ in range(10):
        rmsnorm_kernel.rmsnorm_kernel(input_tensor, weight, eps)

    # Benchmark
    start_time = time.time()
    for _ in range(100):
        rmsnorm_kernel.rmsnorm_kernel(input_tensor, weight, eps)
    end_time = time.time()

    out_kernel = rmsnorm_kernel.rmsnorm_kernel(input_tensor, weight, eps)

    avg_time = (end_time - start_time) / 100
    print(f"CUDA kernel")
    print(f"Average execution time over 100 runs: {avg_time * 1000:.4f} ms")
    print(f"GFLOPS: {(2 * input_tensor.numel()) / (avg_time * 1e9):.2f} GFLOPS")
    print(f"GB/s: {input_tensor.numel() * input_tensor.element_size() * 3 / (avg_time * 1e9):.2f} GB/s")


    rmsnorm = RMSNormTriton(embed_dim=input_tensor.size(-1), eps=eps , weight=weight).to(device=input_tensor.device)
    # Warm-up
    for _ in range(10):
        rmsnorm(input_tensor)
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        rmsnorm(input_tensor)
    end_time = time.time()

    out_triton = rmsnorm(input_tensor)
    avg_time = (end_time - start_time) / 100
    print(f"Triton implementation")
    print(f"Average execution time over 100 runs: {avg_time * 1000:.4f} ms")
    print(f"GFLOPS: {(2 * input_tensor.numel()) / (avg_time * 1e9):.2f} GFLOPS")
    print(f"GB/s: {input_tensor.numel() * input_tensor.element_size() * 3 / (avg_time * 1e9):.2f} GB/s")


    rmsnorm_torch = torch.nn.RMSNorm(normalized_shape=input_tensor.size(-1), eps=eps).to(device=input_tensor.device)
    # Warm-up
    for _ in range(10):
        rmsnorm_torch(input_tensor)
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        rmsnorm_torch(input_tensor)
    end_time = time.time()
    
    out_torch = rmsnorm_torch(input_tensor)
    avg_time = (end_time - start_time) / 100
    print(f"PyTorch implementation")
    print(f"Average execution time over 100 runs: {avg_time * 1000:.4f} ms")
    print(f"GFLOPS: {(2 * input_tensor.numel()) / (avg_time * 1e9):.2f} GFLOPS")
    print(f"GB/s: {input_tensor.numel() * input_tensor.element_size() * 3 / (avg_time * 1e9):.2f} GB/s")


    #accuracy check
    max_diff_triton_torch = torch.max(torch.abs(out_triton - out_torch)).item()
    max_diff_kernel_torch = torch.max(torch.abs(out_kernel - out_torch)).item()
    max_diff_triton_kernel = torch.max(torch.abs(out_triton - out_kernel)).item()

    print(f"Max difference between Triton and PyTorch: {max_diff_triton_torch:.6f}")
    print(f"Max difference between CUDA kernel and PyTorch: {max_diff_kernel_torch:.6f}")
    print(f"Max difference between Triton and CUDA kernel: {max_diff_triton_kernel:.6f}")

if __name__ == "__main__": 
    import torch

    # Example input
    batch_size = 2048
    feature_size = 2048
    input_tensor = torch.randn(batch_size, feature_size, device='cuda' , dtype=torch.float16)
    weight = torch.randn(feature_size, device='cuda' , dtype=torch.float16)

    benchmark_rmsnorm(input_tensor, weight)