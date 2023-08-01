# setup.py
from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            "-gencode=arch=compute_35,code=sm_35",
            "-gencode=arch=compute_37,code=sm_37",
            "-gencode=arch=compute_50,code=sm_50",
            "-gencode=arch=compute_52,code=sm_52",
            "-gencode=arch=compute_53,code=sm_53",
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_62,code=sm_62",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
        sources += sources_cuda
    else:
        raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

setup(
    name='voxel_pooling',
    ext_modules=[
        make_cuda_ext(
            name='voxel_pooling_ext',
            module='ops.voxel_pooling',
            sources=['src/voxel_pooling_forward.cpp'],
            sources_cuda=['src/voxel_pooling_forward_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension})

