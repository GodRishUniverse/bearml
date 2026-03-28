#include "compares.cuh"


namespace simplenet {
    namespace cuda {

        struct EQ_Functor {
            template <typename T>
            __device__ bool operator()(T a, T b) const {
                if constexpr (std::is_floating_point<T>::value) {
                    T eps = T(1e-12);
                    return fabs(a - b) < eps;
                } else {
                    return a == b;
                }
            }
        };

        struct NE_Functor {
            template <typename T>
            __device__ bool operator()(T a, T b) const {
                if constexpr (std::is_floating_point<T>::value) {
                    T eps = T(1e-12);
                    return fabs(a - b) >= eps;
                } else {
                    return a != b;
                }
            }
        };

        struct LE_Functor {
            template <typename T>
            __device__ bool operator()(T a, T b) const {
                return a <= b;
            }
        };

        struct LT_Functor {
            template <typename T>
            __device__ bool operator()(T a, T b) const {
                return a < b;
            }
        };

        struct GT_Functor {
            template <typename T>
            __device__ bool operator()(T a, T b) const {
                return a > b;
            }
        };

        struct GE_Functor {
            template <typename T>
            __device__ bool operator()(T a, T b) const {
                return a >= b;
            }
        };

        template<typename T, typename Op>
        __global__
        void comparison_kernel(T* a, T* b, T* output, size_t size, Op op) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = T(op(a[idx], b[idx]));
            }
        }

        template<typename T>
        void launch_comparison_kernel(T* a, T* b, T* output, size_t size, CompareOp op, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 blockDim(BLOCK_SIZE);
            dim3 gridDim((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
            if (op == CompareOp::LT) {
                comparison_kernel<T, LT_Functor><<<gridDim, blockDim, 0, stream>>>(a, b, output, size, LT_Functor());
            } else if (op == CompareOp::GT) {
                comparison_kernel<T, GT_Functor><<<gridDim, blockDim, 0, stream>>>(a, b, output, size, GT_Functor());
            } else if (op == CompareOp::GE) {
                comparison_kernel<T, GE_Functor><<<gridDim, blockDim, 0, stream>>>(a, b, output, size, GE_Functor());
            } else if (op == CompareOp::LE) {
                comparison_kernel<T, LE_Functor><<<gridDim, blockDim, 0, stream>>>(a, b, output, size, LE_Functor());
            } else if (op == CompareOp::EQ) {
                comparison_kernel<T, EQ_Functor><<<gridDim, blockDim, 0, stream>>>(a, b, output, size, EQ_Functor());
            } else if (op == CompareOp::NE) {
                comparison_kernel<T, NE_Functor><<<gridDim, blockDim, 0, stream>>>(a, b, output, size, NE_Functor());
            } else {
                throw std::invalid_argument("Invalid comparison operator");
            }


            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }


        template void launch_comparison_kernel<__nv_blo>(T* a, T* b, T* output, size_t size, CompareOp op, cudaStream_t stream)

    }
}
