#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17']

nvcc_args = [
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_89,code=compute_89',
]

setup(
    name='channelnorm_cuda',
    ext_modules=[
        CUDAExtension('channelnorm_cuda', [
            'channelnorm_cuda.cc',
            'channelnorm_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
