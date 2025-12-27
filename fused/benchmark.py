import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
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


__global__ void embed_rmsnorm_naive_kernel(
  const int* __restrict__ indices,
  const half2* __restrict__ embedding_matrix,
  const half2* __restrict__ weight_ptr,
  half2* __restrict__ output,
  int num_indices,
  int embedding_dim,
  float eps = 1e-6f
){

  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int index = indices[bx];

  int vcols = embedding_dim / 2;
  
  int embedStart = index * vcols;

  extern __shared__ char shared_mem[];
  half2* smem = reinterpret_cast<half2* >(shared_mem);
  float* smem_partial_sum = reinterpret_cast<float* >(smem+ vcols);

  float partial = 0.0f;
  for(int idx = tid ; idx < vcols ; idx += blockDim.x){
    half2 element = embedding_matrix[embedStart + idx];
    float fx = __half2float(element.x);
    float fy = __half2float(element.y);
    partial += (fx * fx) + (fy * fy);
    smem[idx] = element;
  }

  float total_sum = block_reduce_sum_f32(smem_partial_sum, partial);

  float inv_rms = rsqrtf((total_sum / float(embedding_dim)) + eps);

  for (int idx = tid; idx < vcols; idx += blockDim.x) {
    half2 element = smem[idx];
    half2 w =
        weight_ptr[idx]; 

    float fx = __half2float(element.x) * inv_rms;
    float fy = __half2float(element.y) * inv_rms;

    float out_x = fx * __half2float(w.x);
    float out_y = fy * __half2float(w.y);

    half2 store;
    store.x = __float2half(out_x);
    store.y = __float2half(out_y);

    output[bx * vcols + idx] = store;
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
    int vcols = embed_dim / 2;
    size_t smem_size = (vcols * sizeof(half2)) + (WARP_SIZE * sizeof(float));

    embed_rmsnorm_naive_kernel<<<blocks, threads, smem_size>>>(
        indices.data_ptr<int>(),
        reinterpret_cast<half2*>(embedding_matrix.data_ptr<at::Half>()),
        reinterpret_cast<half2*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half2*>(output.data_ptr<at::Half>()),
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
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_embed_rms'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

# -----------------------------------------------------------------------------
# 2. Benchmarking Utilities
# -----------------------------------------------------------------------------

def benchmark_op(op_func, name, n_iters=100, warmup=10):
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
    print(f"[{name}] Avg execution time: {avg_time_ms:.4f} ms")
    return avg_time_ms

# -----------------------------------------------------------------------------
# 3. Setup & Execution
# -----------------------------------------------------------------------------

# Config for your 4GB GPU
VOCAB_SIZE = 32000
HIDDEN_DIM = 4096  # Must be even
BATCH_SIZE = 32
SEQ_LEN = 128      # Context length
TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN

print(f"Config: Batch={BATCH_SIZE}, Seq={SEQ_LEN}, Hidden={HIDDEN_DIM}, Vocab={VOCAB_SIZE}")
print("Initializing Tensors...")

device = torch.device('cuda')

# Inputs
indices = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int32).flatten()
embedding_table = torch.randn(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=torch.float16)
rms_weight = torch.randn(HIDDEN_DIM, device=device, dtype=torch.float16)

# Outputs (Pre-allocate for fused kernel)
out_fused = torch.empty(TOTAL_TOKENS, HIDDEN_DIM, device=device, dtype=torch.float16)

# PyTorch Baseline Modules
emb_layer = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, _weight=embedding_table)
rms_layer = nn.RMSNorm(HIDDEN_DIM, eps=1e-6).to(device, dtype=torch.float16)
rms_layer.weight.data = rms_weight # Sync weights

# -----------------------------------------------------------------------------
# 4. Correctness Check
# -----------------------------------------------------------------------------
print("\n--- Verifying Correctness ---")

# Run Baseline
ref_out = rms_layer(emb_layer(indices.long()))

# Run Fused
fused_ops.fused_embed_rms(indices, embedding_table, rms_weight, out_fused, 1e-6)

# Compare
diff = (ref_out.view(-1, HIDDEN_DIM) - out_fused).abs().max().item()
print(f"Max Difference: {diff:.6f}")
if diff > 1e-2: # float16 tolerance
    print("❌ FAILED: Implementation mismatch!")
else:
    print("✅ PASSED: Outputs match.")


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
  

model = torch.compile(NN().to(device))

# -----------------------------------------------------------------------------
# 5. Run Benchmark
# -----------------------------------------------------------------------------
print("\n--- Running Benchmark ---")

def run_torch():
    y = emb_layer(indices.long())
    z = rms_layer(y)

def run_fused():
    fused_ops.fused_embed_rms(indices, embedding_table, rms_weight, out_fused, 1e-6)


def run_fused_torch():
   out = model(indices)


t_torch = benchmark_op(run_torch, "PyTorch Native")
t_fused = benchmark_op(run_fused, "Fused Kernel  ")
t_torch_fused = benchmark_op(run_fused_torch , "Compile Torch")
speedup = t_torch / t_fused
speedup_2 = t_torch_fused / t_fused
print(f"\n🚀 Speedup: {speedup:.2f}x") 
print(f"\n🚀 Speedup againt torch.compile: {speedup_2:.2f}x") 