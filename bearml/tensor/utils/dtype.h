#pragma once
#include <cstdint>
#include <string_view>
#include <type_traits>

#if __has_include(<stdfloat>)
    #include <stdfloat>
#endif

#if defined(BEARML_USE_CUDA)
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

namespace bearml {
    // will add FP8 and MXFP4 later - TODO
    enum class DType : uint8_t { F64, F32, F16, F8, MXFP4, BF16, I64, I32, I16, I8 };

    namespace detail {
        template <typename T> struct dtype_unsupported : std::false_type {};

        template <typename T>
        struct dtype_of_helper {
            static_assert(dtype_unsupported<T>::value, "dtype_of: unsupported element type");
        };

        template <> struct dtype_of_helper<double>  { static constexpr DType value = DType::F64; };
        template <> struct dtype_of_helper<float>   { static constexpr DType value = DType::F32; };
        template <> struct dtype_of_helper<int64_t> { static constexpr DType value = DType::I64; };
        template <> struct dtype_of_helper<int32_t> { static constexpr DType value = DType::I32; };
        // TODO: F16 needs its own host type (std::float16_t) distinct from int16_t
        template <> struct dtype_of_helper<int16_t> { static constexpr DType value = DType::I16; };
        template <> struct dtype_of_helper<int8_t>  { static constexpr DType value = DType::I8; };

        #if defined(__STDCPP_BFLOAT16_T__)
            template <> struct dtype_of_helper<std::bfloat16_t> { static constexpr DType value = DType::BF16; };
        #endif

        #if defined(BEARML_USE_CUDA)
            template <> struct dtype_of_helper<__nv_bfloat16> { static constexpr DType value = DType::BF16; };
            template <> struct dtype_of_helper<__half>        { static constexpr DType value = DType::F16; };
        #endif
    }

    template <typename T>
    inline constexpr DType dtype_of = detail::dtype_of_helper<T>::value;

    constexpr std::string_view dtype_name(DType d) {
        switch (d) {
            case DType::F64:   return "float64";
            case DType::F32:   return "float32";
            case DType::F16:   return "float16";
            case DType::F8:    return "float8";
            case DType::MXFP4: return "mxfp4";
            case DType::BF16:  return "bfloat16";
            case DType::I64:   return "int64";
            case DType::I32:   return "int32";
            case DType::I16:   return "int16";
            case DType::I8:    return "int8";
        }
        return "unknown";
    }

    constexpr bool is_floating(DType d) {
        switch (d) {
            case DType::F64:
            case DType::F32:
            case DType::F16:
            case DType::F8:
            case DType::MXFP4:
            case DType::BF16:
                return true;
            default:
                return false;
        }
    }
}
