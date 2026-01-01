#include "../../../includes/helper.h"
#include "../../../includes/ops.h"


// __restrict__ keyword usage: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
namespace simplenet {
    namespace cuda {
        // restrict is an optimization measure that basically tells the compiler that these are separate data


        // TODO: Cache frequently accessed data in shared memory
        // CASE: BROADCASTED VERSION
        template <typename T>
        __global__
        void element_wise_broadcast(
            const size_t* __restrict__ strides_a,
            const size_t* __restrict__ strides_b,
            const size_t* __restrict__ res_shape,
            size_t res_flat_shape,
            size_t outShapeSize,
            const T* __restrict__ a,
            const T* __restrict__ b,
            T* res,
            uint16_t op_code
        ){
            size_t thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

            if (thread_idx< res_flat_shape){
                size_t tmp = thread_idx;
                size_t offA = 0;
                size_t offB = 0;

                for (int d = outShapeSize - 1; d >= 0; --d) {
                    size_t coord  = tmp % res_shape[d];
                    tmp /= res_shape[d];
                    offA += coord * strides_a[d];
                    offB += coord * strides_b[d];
                }

                T result;
                switch (op_code) {
                    case OP_ADD: {
                        result = a[offA]+b[offB];
                        break;
                    }
                    case OP_SUB:{
                        result = a[offA]-b[offB];
                        break;
                    }
                    case OP_MUL:{
                        result = a[offA]*b[offB];
                        break;
                    }
                    case OP_DIV:{
                        result = a[offA]/b[offB];
                        break;
                    }
                    // default operation -> since we cannot throw an error inside the kernel
                    default:
                        result = static_cast<T>(NAN);
                        break;
                }
                res[thread_idx] = result;
            }
        }

        // CASE: NO BROADCASTING NEEDED
        template <typename T>
        __global__
        void element_wise_contiguous(
            const T* __restrict__ a,
            const T* __restrict__ b,
            T* res,
            size_t n,
            uint16_t op_code
        ){
            size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (thread_idx < n) {
                T result;
                switch (op_code) {
                    case OP_ADD: {
                        result = a[thread_idx]+b[thread_idx];
                        break;
                    }
                    case OP_SUB:{
                        result = a[thread_idx]-b[thread_idx];
                        break;
                    }
                    case OP_MUL:{
                        result = a[thread_idx]*b[thread_idx];
                        break;
                    }
                    case OP_DIV:{
                        result = a[thread_idx]/b[thread_idx];
                        break;
                    }
                    // default operation -> since we cannot throw an error inside the kernel
                    default:
                        result = static_cast<T>(NAN);
                        break;
                }
                res[thread_idx] = result;
            }
        }
    }
}
