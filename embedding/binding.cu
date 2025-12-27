#include <torch/extension.h>

#include "kernels/better_embed.cuh"

torch::Tensor embedding_kernel(
  torch::Tensor index,
  torch::Tensor weight,
  int N,
  int embed_dim
){
  if(!index.is_contiguous()) index = index.contiguous();
  if(!weight.is_contiguous()) weight = weight.contiguous();

  auto output = torch::empty({N , embed_dim});

  const int* idx_ptr = reinterpret_cast<const int*>(index.data_ptr());
  float* weight_ptr = reinterpret_cast<float* >(weight.data_ptr());
  float* output_ptr = reinterpret_cast<float* >(output.data_ptr());
  launch_vec_embedding(
    idx_ptr,
    weight_ptr,
    output_ptr,
    N , embed_dim
  );


  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
  }

  return output;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("rmsnorm_kernel_vec", &fused_rmsnorm, "Fused RMSNorm (BF16)");
    m.def("embedding_kernel", &embedding_kernel, "Fused RMSNorm (BF16)");
}