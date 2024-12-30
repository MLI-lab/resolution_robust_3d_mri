
#include <torch/extension.h>

#include "rasterizer_lib.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize", &rasterize_gaussians_cuda_complex);
    m.def("rasterize_backward", &rasterize_gaussians_cuda_complex_backward);
}