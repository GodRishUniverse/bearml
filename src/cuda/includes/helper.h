#pragma once
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>


#ifndef CUDA_HELPER
#define CUDA_HELPER


namespace simplenet {
    namespace cuda {
        // helper
        inline dim3 get_blocks(size_t n, size_t threads= 256) {
            return dim3((n + threads - 1) / threads);
        }

    }
}
#endif
