#pragma once
#if defined(BEARML_USE_CUDA)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>

    // the custom types - CUDA does not support <stdfloat> because that is a C++23 thing and NVCC does not support C++23
    #include <cuda_fp16.h>
    #include <cuda_fp8.h>
    #include <cuda_bf16.h>
#endif
#include "cuda_consts.h" // we will have all our constants here
#include "cuda_cas_traits.h" // has Copy-and-Swap traits used for atomic operations

#if defined(BEARML_USE_CUDA)
#include "device_atomic_functions.h"  // CUDA SDK header
#endif
#include "../../operators/ops.h"
#include "../../operators/padding_ops.h"
