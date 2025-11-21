from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# Force CUDA check to look at env var if torch fails to detect it during build
# But actually, we want to allow building without CUDA if necessary (CPU mode fallback or just build phase)
# However, MASt3R-SLAM *needs* the backend extensions for functionality.
# If we are building in Docker with TORCH_CUDA_ARCH_LIST set, we can force compilation even if runtime CUDA isn't there.

# Fix: Initialize ext_modules
ext_modules = []

# Check if we should force CUDA build (e.g. in Docker)
force_cuda = os.getenv("FORCE_CUDA", "0") == "1" or os.getenv("TORCH_CUDA_ARCH_LIST") is not None
has_cuda = torch.cuda.is_available() or force_cuda

include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")
    extra_compile_args["nvcc"] = [
        "-O3",
        # Architecture flags are often handled by TORCH_CUDA_ARCH_LIST, but keeping these as fallback
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
    ]
    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend! (MASt3R-SLAM will likely fail)")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
