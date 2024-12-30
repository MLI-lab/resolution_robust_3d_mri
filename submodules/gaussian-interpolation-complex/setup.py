from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="gaussian_rasterizer_complex",
    packages=['gaussian_rasterizer_complex'],
    ext_modules=[
        CUDAExtension(
            name="gaussian_rasterizer_complex._C",
            sources=[
            "rasterizer_complex_per_tile/forward.cu",
            "rasterizer_complex_per_tile/backward.cu",
            "rasterizer_complex_per_tile/rasterizer_impl.cu",
            "aux.cu",
            "rasterizer_lib.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
