#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace simplenet {
    namespace cuda {
        // here basically we define what kind of int type to use for atomicCAS - as it only supports unsigned int/long long for 32/64-bit types or shorts
        template <typename T> struct CASTraits;

        // 16-bit types — atomicCAS supports unsigned short on sm_70+ architecture - flag indicates a NVIDIA GPU with Compute Capability 7.0
        template <> struct CASTraits<__nv_bfloat16> {
            using int_type = unsigned short;
        };
        template <> struct CASTraits<__half> {
            using int_type = unsigned short;
        };

        // 32-bit types
        template <> struct CASTraits<float> {
            using int_type = unsigned int;
        };
        template <> struct CASTraits<int32_t> {
            using int_type = unsigned int;
        };

        // 64-bit types
        template <> struct CASTraits<double> {
            using int_type = unsigned long long;
        };
        template <> struct CASTraits<int64_t> {
            using int_type = unsigned long long;
        };

        // 16-bit integer
        template <> struct CASTraits<int16_t> {
            using int_type = unsigned short;
        };

        // int8_t has NO atomicCAS support - we handle via specialization (TODO: need to think about it)
    }
}
