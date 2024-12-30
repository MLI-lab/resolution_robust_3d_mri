
#pragma once

#include <functional>
#include <torch/torch.h>

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t);

struct BinningState {
    size_t sorting_size; // unsigned long
    uint32_t* point_list_keys_unsorted;
    uint32_t* point_list_keys;
    uint32_t* point_list_unsorted;
    uint32_t* point_list;
    char* list_sorting_space;
		
    static BinningState fromChunk(char*& chunk, size_t P);
};

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

template<typename T> 
size_t required(size_t P)
{
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}

uint64_t getHigherMsb(uint64_t n);