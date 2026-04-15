#include "accumulate.cuh"

namespace simplenet {
    namespace cuda {

        template <typename T>
        void launch_accumulate_kernel(T *d_data, T *d_out, std::vector<int> data_shape, std::vector<int> out_shape, int64_t size, int64_t offset_new_shape, int64_t offset_old, reductions::ReductionOps op,  bool keepdims, cudaStream_t stream) {
            // TODO - launch appropriate reduction kernel depending on op
            switch (op) {
                case reductions::ReductionOps::SUM:
                    simplenet::cuda::launch_sum_acc_kernel<T>(d_data, d_out, offset_new_shape, offset_old, size, stream);
                    break;
                case reductions::ReductionOps::MEAN:
                case reductions::ReductionOps::MAX:
                case reductions::ReductionOps::MIN:
                case reductions::ReductionOps::PROD:
                case reductions::ReductionOps::ARG_MAX:
                case reductions::ReductionOps::ARG_MIN:
                    throw std::runtime_error("REDUCTION OP NOT IMPLEMENTED YET");
                default:
                    throw std::runtime_error("Unsupported reduction op");
            }
        }

        // floats
        INSTANTIATE_ACCUMULATE(__nv_bfloat16);
        INSTANTIATE_ACCUMULATE(__half);
        INSTANTIATE_ACCUMULATE(float);
        INSTANTIATE_ACCUMULATE(double);

        // ints
        INSTANTIATE_ACCUMULATE(int32_t);

    }
}
