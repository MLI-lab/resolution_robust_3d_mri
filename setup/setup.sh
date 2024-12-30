#!/bin/bash
apt-get update && apt-get install -y make gcc g++ libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev wget nvidia-cuda-gdb libsparsehash-dev

# install bart
mkdir -p libs
cd libs
wget https://github.com/mrirecon/bart/archive/refs/tags/v0.6.00.tar.gz
tar -xf v0.6.00.tar.gz
cd bart-0.6.00 && make
cd ../..

# wheels for depthwisedm
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install "git+https://github.com/samb-t/torchsparse.git"

# compile local cuda kernels for gaussian interpolation
pip install -e submodules/simple-knn
pip install -e submodules/gaussian-interpolation-complex