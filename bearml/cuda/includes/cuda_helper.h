#pragma once
// Plain C++ standard headers used pervasively by the kernel layer.
// Kept OUTSIDE the CUDA guard and unconditional: they are valid in both
// GPU and CPU-only builds, and the .cu translation units historically
// rely on cuda_helper.h pulling these in.
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <functional>

#include "cuda_imports.h" // all cuda imports

#if defined(BEARML_USE_CUDA)

    #ifndef CUDA_HELPER
    #define CUDA_HELPER


    namespace bearml {
        namespace cuda {
            // helper
            inline dim3 get_blocks(size_t n, size_t threads= 256) {
                return dim3((n + threads - 1) / threads);
            }

        }
    }
    #endif


    // Macro functions -> https://stackoverflow.com/questions/163365/how-do-i-make-a-c-macro-behave-like-a-function
    // error checking macro
    #ifndef CUDA_CHECK_MACRO
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                throw std::runtime_error( \
                    std::string("CUDA error at ") + __FILE__ + ":" + \
                    std::to_string(__LINE__) + " - " + \
                    cudaGetErrorString(error)); \
            } \
        } while(0)


    #endif

#endif
