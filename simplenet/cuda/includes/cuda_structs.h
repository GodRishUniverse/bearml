#pragma once
#include <cstddef>

#ifndef CUDA_STRUCTS_H
#define CUDA_STRUCTS_H

// All broadcast parameters - leave (some-irrelevant) data empty for the different kernel launches if that data is not needed
template<typename T>
struct BroadcastParams {
    const T* a;
    const T* b;
    const T* res;

    int* strides_a;
    int* strides_b;
    int* res_shape;
    size_t strides_a_size;
    size_t strides_b_size;
    size_t res_shape_size;
    size_t num_dims;
    size_t res_flat_shape;
};

#endif
