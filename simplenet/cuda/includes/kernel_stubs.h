#pragma once
// CPU-only build stubs (SIMPLENET_USE_CUDA undefined).
//
// Every cuda::launch_*
// These stubs exist only so those dead branches compile; if one is ever
// reached at runtime it throws a clear error instead of silently misbehaving.

#include <stdexcept>

namespace simplenet {
    namespace cuda {

    [[noreturn]] inline void cuda_not_built() {
        throw std::runtime_error(
            "A CUDA kernel was invoked but SimpleNet was built without CUDA "
            "(SIMPLENET_USE_CUDA=OFF). Rebuild with CUDA enabled to use GPU ops.");
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
    #define SIMPLENET_CUDA_STUB(name)                                   \
        template <typename... Expl, typename... Args>                   \
        CudaUnavailable name(Args&&...) { cuda_not_built(); }

    SIMPLENET_CUDA_STUB(launch_elementwise_contiguous)
    SIMPLENET_CUDA_STUB(launch_elementwise_broadcast)
    SIMPLENET_CUDA_STUB(launch_elementwise_contiguous_with_constant)
    SIMPLENET_CUDA_STUB(launch_elementwise_unary)
    SIMPLENET_CUDA_STUB(launch_gemm_contiguous)
    SIMPLENET_CUDA_STUB(launch_gemm_broadcasted)
    SIMPLENET_CUDA_STUB(launch_check_equal_kernel)
    SIMPLENET_CUDA_STUB(launch_check_zero_kernel)
    SIMPLENET_CUDA_STUB(launch_comparison_kernel)
    SIMPLENET_CUDA_STUB(launch_accumulate_kernel)
    SIMPLENET_CUDA_STUB(launch_sum_kernel)
    SIMPLENET_CUDA_STUB(launch_fill)
    SIMPLENET_CUDA_STUB(launch_linspace_kernel)
    SIMPLENET_CUDA_STUB(launch_transpose_kernel)
    SIMPLENET_CUDA_STUB(launch_concat_kernel)
    SIMPLENET_CUDA_STUB(launch_padd_with_constant)
    SIMPLENET_CUDA_STUB(launch_sign_contiguous)

    namespace utils {
        SIMPLENET_CUDA_STUB(launch_dtype_change)
        SIMPLENET_CUDA_STUB(launch_contiguous_gather)
    }

    #undef SIMPLENET_CUDA_STUB

    }  // namespace cuda
}  // namespace simplenet
