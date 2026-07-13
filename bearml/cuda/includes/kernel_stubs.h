#pragma once
// CPU-only build stubs (BEARML_USE_CUDA undefined).
//
// Every cuda::launch_*
// These stubs exist only so those dead branches compile; if one is ever
// reached at runtime it throws a clear error instead of silently misbehaving.

#include <stdexcept>

namespace bearml {
    namespace cuda {

    [[noreturn]] inline void cuda_not_built() {
        throw std::runtime_error(
            "A CUDA kernel was invoked but BearML was built without CUDA "
            "(BEARML_USE_CUDA=OFF). Rebuild with CUDA enabled to use GPU ops.");
    }

    // Implicitly converts to any type the call site expects (e.g.
    // `return cuda::launch_check_zero_kernel(...)` where a bool is needed).
    // Never actually produces a value — the stub throws first.
    struct CudaUnavailable {
        template <typename U>
        operator U() const { cuda_not_built(); }
    };

    // Two parameter packs: explicit template args at call sites (e.g.
    // launch_fill<T>(...), launch_dtype_change<T,T2>(...)) are absorbed by
    // `Expl`, while the real arguments are deduced into `Args` — so forwarding
    // references bind correctly regardless of the explicit <...> args.
    #define BEARML_CUDA_STUB(name)                                   \
        template <typename... Expl, typename... Args>                   \
        CudaUnavailable name(Args&&...) { cuda_not_built(); }

    BEARML_CUDA_STUB(launch_elementwise_contiguous)
    BEARML_CUDA_STUB(launch_elementwise_broadcast)
    BEARML_CUDA_STUB(launch_elementwise_contiguous_with_constant)
    BEARML_CUDA_STUB(launch_elementwise_unary)
    BEARML_CUDA_STUB(launch_gemm_contiguous)
    BEARML_CUDA_STUB(launch_gemm_broadcasted)
    BEARML_CUDA_STUB(launch_check_equal_kernel)
    BEARML_CUDA_STUB(launch_check_zero_kernel)
    BEARML_CUDA_STUB(launch_comparison_kernel)
    BEARML_CUDA_STUB(launch_accumulate_kernel)
    BEARML_CUDA_STUB(launch_sum_kernel)
    BEARML_CUDA_STUB(launch_softmax_kernel)
    BEARML_CUDA_STUB(launch_fill)
    BEARML_CUDA_STUB(launch_linspace_kernel)
    BEARML_CUDA_STUB(launch_transpose_kernel)
    BEARML_CUDA_STUB(launch_concat_kernel)
    BEARML_CUDA_STUB(launch_padd_with_constant)
    BEARML_CUDA_STUB(launch_sign_contiguous)

    namespace utils {
        BEARML_CUDA_STUB(launch_dtype_change)
        BEARML_CUDA_STUB(launch_contiguous_gather)
    }

    #undef BEARML_CUDA_STUB

    }  // namespace cuda
}  // namespace bearml
