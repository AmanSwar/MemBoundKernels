from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension , BuildExtension

setup(
  name='rmsnorm_cuda',
  ext_modules=[
    CUDAExtension(
      "rmsnorm_kernel",
      sources=["binding.cu"],
      extra_compile_args={
          "cxx": [],
          "nvcc": [
              "-O3",
              "-gencode",
              "arch=compute_80,code=sm_80",
              "--expt-relaxed-constexpr",
          ],
      },
  ),
  ],
  cmdclass={"build_ext": BuildExtension},
)