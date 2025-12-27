import torch
import torch.nn as nn

def benchmark(M, N):
    # 1. Setup
    input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float16)
    rmsnorm = nn.RMSNorm(normalized_shape=N, eps=1e-6).to(device='cuda', dtype=torch.float16)
    
    # 2. Warmup (Get the JIT/Allocation overhead out of the way)
    for _ in range(10):
        _ = rmsnorm(input_tensor)
    torch.cuda.synchronize()

    # 3. Timing with CUDA Events (High Precision)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):  # Run enough iterations to average out noise
        _ = rmsnorm(input_tensor)
    end_event.record()
    
    # WAIT for GPU to finish
    torch.cuda.synchronize()
    
    # Elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / 100
    avg_time_s = avg_time_ms / 1000

    print(f"Benchmark for RMSNorm ({M}, {N})")
    print(f"Time: {avg_time_ms:.4f} ms")

    # 4. Correct Byte Math
    # Read (Input) + Write (Output) + Read (Weight - negligible but technical)
    total_bytes = (M * N * 2) + (M * N * 2) + 2 * M
    
    # GB/s = (Bytes / 1e9) / Seconds
    gbs = (total_bytes / 1e9) / avg_time_s
    print(f"GB/s: {gbs:.2f} GB/s")
    print("-" * 30)

def main():
    N = 2048
    # Test larger sizes to saturate the GPU
    for i in range(1, 11):
        M = 128*8 * i 
        benchmark(M, N)

if __name__ == "__main__":
    main()