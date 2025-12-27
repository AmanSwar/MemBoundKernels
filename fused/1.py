import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CUDA Source Code (Corrected & Wrappers Added)
# -----------------------------------------------------------------------------
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define WARP_SIZE 32

// --- Helper Functions ---

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  const unsigned int MASK = 0xffffffffu;
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset >= 1; offset >>= 1) {
    v += __shfl_xor_sync(MASK, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_f32(float *sdata, float val) {
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int wid = tid / WARP_SIZE; // warp id

  val = warp_reduce_sum_f32(val);

  if (lane == 0)
    sdata[wid] = val;

  __syncthreads();

  float total = 0.0f;
  if (wid == 0) {
    float v = (lane < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? sdata[lane]
                                                                  : 0.0f;
    v = warp_reduce_sum_f32(v);
    if (lane == 0)
      sdata[0] = v;
  }

  __syncthreads();
  total = sdata[0];
  return total;
}

#define LD32BITS(value) (reinterpret_cast<const half2 *>(&(value))[0])
#define ST32BITS(addr, value) (reinterpret_cast<half2 *>(&(addr))[0] = (value))

__global__ void __launch_bounds__(256,2)
embed_rmsnorm_vec_kernel(
  const int* __restrict__ indices,
  const half* __restrict__ embedding_matrix,
  const half* __restrict__ weight_ptr,
  half* __restrict__ output,
  int num_indices,
  int embedding_dim,
  float eps = 1e-6f
){
  int tid = threadIdx.x;
  int bx = blockIdx.x;
  
  int index = indices[bx];
  int vcols = (embedding_dim + 1) / 2;
  
  int embedStart = index * embedding_dim;
  
  extern __shared__ char shared_mem[];
  half2* smem = reinterpret_cast<half2*>(shared_mem);
  float* smem_partial_sum = reinterpret_cast<float*>(smem + vcols);
  
  float partial = 0.0f;
  int full_quads = embedding_dim / 4;
  bool has_tail = (embedding_dim % 4) != 0;
  
  // Process 4 elements (2 half2s) at a time
  for(int idx = tid; idx < full_quads; idx += blockDim.x){
    half2 element1 = LD32BITS(embedding_matrix[embedStart + idx * 4]);
    float fx1 = __half2float(element1.x);
    float fy1 = __half2float(element1.y);
    
    half2 element2 = LD32BITS(embedding_matrix[embedStart + idx * 4 + 2]);
    float fx2 = __half2float(element2.x);
    float fy2 = __half2float(element2.y);
    
    partial += fx1 * fx1 + fy1 * fy1 + fx2 * fx2 + fy2 * fy2;
    
    smem[idx * 2] = element1;
    smem[idx * 2 + 1] = element2;
  }
  
  // Handle tail elements when embedding_dim % 4 != 0
  if(has_tail && tid == 0){
    int tail_idx = full_quads * 2; 
    int tail_start = embedStart + full_quads * 4;
    int tail_count = embedding_dim % 4;
    
    if(tail_count == 1){
      half tail_element = embedding_matrix[tail_start];
      float fx = __half2float(tail_element);
      partial += fx * fx;
      
      half2 new_el;
      new_el.x = tail_element;
      new_el.y = __float2half(0.0f);
      smem[tail_idx] = new_el;
      
    } else if(tail_count == 2){
      half2 tail_element = LD32BITS(embedding_matrix[tail_start]);
      float fx = __half2float(tail_element.x);
      float fy = __half2float(tail_element.y);
      partial += fx * fx + fy * fy;
      
      smem[tail_idx] = tail_element;
      
    } else if(tail_count == 3){
      half2 tail_element = LD32BITS(embedding_matrix[tail_start]);
      float fx1 = __half2float(tail_element.x);
      float fy1 = __half2float(tail_element.y);
      
      half tail_element2 = embedding_matrix[tail_start + 2];
      float fx2 = __half2float(tail_element2);
      
      partial += fx1 * fx1 + fy1 * fy1 + fx2 * fx2;
      
      smem[tail_idx] = tail_element;
      half2 last_el;
      last_el.x = tail_element2;
      last_el.y = __float2half(0.0f);
      smem[tail_idx + 1] = last_el;
    }
  }
  
  __syncthreads();
  
  float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);
  
  float inv_rms = rsqrtf((total_sum / float(embedding_dim)) + eps);
  
  int output_start = bx * embedding_dim;
  
  // Apply normalization and weights with vectorized access
  for(int idx = tid; idx < vcols; idx += blockDim.x){
    half2 element = smem[idx];
    half2 w = LD32BITS(weight_ptr[idx * 2]);
    
    float fx = __half2float(element.x) * inv_rms;
    float fy = __half2float(element.y) * inv_rms;
    
    float out_x = fx * __half2float(w.x);
    float out_y = fy * __half2float(w.y);
    
    half2 out_el;
    out_el.x = __float2half(out_x);
    out_el.y = __float2half(out_y);
    
    ST32BITS(output[output_start + idx * 2], out_el);
  }
}

// --- C++ Wrapper ---
void fused_embed_rms(
    torch::Tensor indices,
    torch::Tensor embedding_matrix,
    torch::Tensor weight,
    torch::Tensor output,
    float eps
) {
    int num_indices = indices.numel();
    int embed_dim = embedding_matrix.size(1);
    
    int threads = 256;
    int blocks = num_indices;
    int vcols = (embed_dim + 1) / 2;
    size_t smem_size = (vcols * sizeof(half2)) + (256 * sizeof(float));

    embed_rmsnorm_vec_kernel<<<blocks, threads, smem_size>>>(
        indices.data_ptr<int>(),
        reinterpret_cast<half*>(embedding_matrix.data_ptr<at::Half>()),
        reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        num_indices,
        embed_dim,
        eps
    );
  }
'''

cpp_source = r'''
void fused_embed_rms(
    torch::Tensor indices,
    torch::Tensor embedding_matrix,
    torch::Tensor weight,
    torch::Tensor output,
    float eps
);
'''

# Compile the extension
print("Compiling CUDA extension...")
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_embed_rms'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)
print("✅ CUDA extension compiled successfully!\n")

# -----------------------------------------------------------------------------
# 2. Benchmarking Utilities
# -----------------------------------------------------------------------------

class NN(nn.Module):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__()
    self.VOCAB_SIZE = 32000
    self.EMBED_DIM = 2048
    
    self.embedding = nn.Embedding(self.VOCAB_SIZE , self.EMBED_DIM)
    self.norm = nn.RMSNorm(self.EMBED_DIM)

  def forward(self , x):

    out = self.norm(self.embedding(x))
    return out

model = torch.compile(NN().to(torch.device('cuda')))

def benchmark_op(op_func, n_iters=100, warmup=10):
    """Benchmark an operation and return average time in ms"""
    # Warmup
    for _ in range(warmup):
        op_func()
    torch.cuda.synchronize()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_iters):
        op_func()
    end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / n_iters
    return avg_time_ms

def verify_correctness(indices, embedding_table, rms_weight, out_fused, 
                       emb_layer, rms_layer, verbose=True):
    """Verify that fused kernel matches PyTorch baseline"""
    # Run Baseline
    ref_out = rms_layer(emb_layer(indices.long()))
    
    # Run Fused
    fused_ops.fused_embed_rms(indices, embedding_table, rms_weight, out_fused, 1e-6)
    
    # Compare
    diff = (ref_out.view(-1, embedding_table.size(1)) - out_fused).abs().max().item()
    
    if verbose:
        print(f"Max Difference: {diff:.6f}")
        if diff > 1e-2:  # float16 tolerance
            print("❌ FAILED: Implementation mismatch!")
            return False
        else:
            print("✅ PASSED: Outputs match.")
            return True
    
    return diff <= 1e-2

# -----------------------------------------------------------------------------
# 3. Comprehensive Benchmark
def run_comprehensive_benchmark():
    # Fixed configuration
    VOCAB_SIZE = 32000
    HIDDEN_DIM = 4096  # Must be even
    SEQ_LEN = 128      # Context length
    
    # Batch sizes to test
    BATCH_SIZES = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
    
    print("=" * 80)
    print("COMPREHENSIVE FUSED EMBEDDING + RMSNorm BENCHMARK")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Seq Length: {SEQ_LEN}")
    print("=" * 80)
    
    device = torch.device('cuda')
    
    # Initialize shared tensors
    embedding_table = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=torch.float16)
    rms_weight = torch.randn(HIDDEN_DIM, device=device, dtype=torch.float16)
    
    # PyTorch Baseline Modules
    emb_layer = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, _weight=embedding_table)
    rms_layer = nn.RMSNorm(HIDDEN_DIM, eps=1e-6).to(device, dtype=torch.float16)
    rms_layer.weight.data = rms_weight
    
    results = []
    
    # --- Verifying Correctness ---
    # (Keeping your existing verification logic briefly here for safety)
    print("\n--- Verifying Correctness (Batch Size = 8) ---")
    indices = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN), device=device, dtype=torch.int32).flatten()
    out_fused = torch.empty(8 * SEQ_LEN, HIDDEN_DIM, device=device, dtype=torch.float16)
    is_correct = verify_correctness(indices, embedding_table, rms_weight, out_fused, 
                                    emb_layer, rms_layer, verbose=True)
    if not is_correct: return

    print("\n" + "=" * 80)
    print(f"{'Batch':>6} {'Time (ms)':>12} {'Throughput (GB/s)':>20} {'Speedup':>10}")
    print("=" * 80)
    
    # Run benchmarks
    for batch_size in BATCH_SIZES:
        TOTAL_TOKENS = batch_size * SEQ_LEN
        
        # 1. Calculate Data Volume (Bytes) for Throughput
        # Read Indices: (B*S * 4 bytes)
        # Read Rows:    (B*S * H * 2 bytes)
        # Write Output: (B*S * H * 2 bytes)
        # We ignore RMS weights read (H*2) as it's negligible and cached
        total_bytes = (TOTAL_TOKENS * 4) + (TOTAL_TOKENS * HIDDEN_DIM * 2) + (TOTAL_TOKENS * HIDDEN_DIM * 2)
        total_gb = total_bytes / 1e9
        
        # Tensors
        indices = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=device, dtype=torch.int32).flatten()
        out_fused = torch.empty(TOTAL_TOKENS, HIDDEN_DIM, device=device, dtype=torch.float16)
        
        # Define closures
        def run_torch():
            y = emb_layer(indices.long())
            z = rms_layer(y)
        
        def run_fused():
            fused_ops.fused_embed_rms(indices, embedding_table, rms_weight, out_fused, 1e-6)
        
        def torch_compiled():
            y = model(indices.long())
        
        # Measure
        t_torch = benchmark_op(run_torch)
        t_fused = benchmark_op(run_fused)
        t_compiled = benchmark_op(torch_compiled)
        
        # Metrics
        gbps_fused = total_gb / (t_fused / 1000.0)
        gbps_torch = total_gb / (t_torch / 1000.0)
        gbps_compiled = total_gb / (t_compiled / 1000.0)
        speedup = t_torch / t_fused
        
        print(f"{batch_size:>6} {t_fused:>12.4f} {gbps_fused:>20.2f} {speedup:>10.2f}x")
        
        results.append({
            'batch': batch_size,
            't_torch': t_torch,
            't_fused': t_fused,
            't_compiled': t_compiled,
            'gbps_torch': gbps_torch,
            'gbps_fused': gbps_fused,
            'gbps_compiled': gbps_compiled
        })

    # -----------------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------------
    try:
        # plt.style.use('dark_background') # Optional: looks cooler
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = [r['batch'] for r in results]
        
        # PLOT 1: Latency (ms) - Lower is better
        ax1.plot(x, [r['t_torch'] for r in results], 'o--', color='#ff7f0e', label='PyTorch Eager')
        ax1.plot(x, [r['t_compiled'] for r in results], '^--', color='#2ca02c', label='Torch Compile')
        ax1.plot(x, [r['t_fused'] for r in results], 's-', color='#1f77b4', linewidth=2, label='Fused Kernel')
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Kernel Execution Time (Lower is Better)')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # PLOT 2: Throughput (GB/s) - Higher is better
        ax2.plot(x, [r['gbps_torch'] for r in results], 'o--', color='#ff7f0e', label='PyTorch Eager')
        ax2.plot(x, [r['gbps_compiled'] for r in results], '^--', color='#2ca02c', label='Torch Compile')
        ax2.plot(x, [r['gbps_fused'] for r in results], 's-', color='#1f77b4', linewidth=2, label='Fused Kernel')
        
        # Add Theoretical Peak line (Optional - adjust based on your GPU, e.g., A100 ~1555 GB/s)
        # ax2.axhline(y=1555, color='r', linestyle=':', alpha=0.5, label='A100 Peak Mem BW')

        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (GB/s)')
        ax2.set_title('Memory Throughput (Higher is Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig('fused_ops_benchmark.png', dpi=150)
        print("\n📊 Saved plot to 'fused_ops_benchmark.png'")
        plt.show()
        
    except ImportError:
        print("Skipping plot (matplotlib not found)")

    return results

if __name__ == "__main__":
    run_comprehensive_benchmark()#