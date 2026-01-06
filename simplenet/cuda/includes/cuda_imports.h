#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// the custom types - CUDA does not support <stdfloat> because that is a C++23 thing and NVCC does not support C++23
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "cuda_structs.h" // we include our structs.h here cause we now include the helper directly
