#pragma once

#include "../includes/cuda_helper.h"
#include <cstdint>
#include "../../tensor/utils/compare_ops.h"

namespace bearml {
    namespace cuda {


        template<typename T>
        void launch_comparison_kernel(T* a, T* b, T* output, size_t size, CompareOp op, cudaStream_t stream= nullptr);

    }
}
