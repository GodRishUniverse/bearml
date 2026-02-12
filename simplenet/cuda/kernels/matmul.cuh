#include "../includes/cuda_helper.h"

namespace simplenet {
    namespace cuda {

        template<typename T>
        void launch_gemm_broadcasted(
            T* d_a,
            T* d_b,
            T* d_c,
            int m,
            int k,
            int n,
            T alpha,
            T beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );

        template<typename T>
        void launch_gemm_contiguous(
            T* d_a,
            T* d_b,
            T* d_c,
            int batchsize,
            int m,
            int n,
            int k,
            T alpha,
            T beta,
            cudaStream_t stream
        );

    }
}
