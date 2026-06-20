#include "softmax.cuh"
#include "sum.cuh"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace simplenet {
    namespace cuda {

        // todo: think about how dim is used in softmax (I know per row is naive and default but generalization is good)
        // todo: another problem is when the shared memory overlaps as not all data items on a CU/SM
        // figure out how to handle this overlap
        template<typename T>
        __global__ void softmax_kernel(const T* d_data, T* d_out,  int* shape, int shape_size, int* stride, int stride_size, int dim, int64_t size) {
            int shape_dim = shape[dim];
            int stride_dim = stride[dim];

            // shared memory for cumulative sum  = number of rows basically
            // example is like 4*5 will need 1 float per row (4 floats)
            // so shape[dim] * sizeof(T) is the size of the shared memory per block
            extern __shared__ T cum_sum_storage[]; // needs to be declared outside the kernel in the <<<...>>> call with the shared memory size (has to be shape[dim] * sizeof(T))

            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                // one thread per element
                T temp = exp(d_data[idx]);
                d_out[idx] = temp;
                // idx is the global index, we use it to index into the shared memory
                int local_idx = ceil(idx / shape_dim);
                cum_sum_storage[local_idx] += temp;

            }
            __syncthreads(); // I was thinking of using a shared memory reduction to do a cumulative sum and then divide by the sum

            // apply the divide after the cumulative sum is computed
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                // one thread per element

            }
        }


        // floats
        INSTANTIATE_SOFTMAX(__nv_bfloat16);
        INSTANTIATE_SOFTMAX(__half);
        INSTANTIATE_SOFTMAX(float);
        INSTANTIATE_SOFTMAX(double);

        // ints
        // INSTANTIATE_SOFTMAX(int8_t);
        // INSTANTIATE_SOFTMAX(int16_t);
        INSTANTIATE_SOFTMAX(int32_t);
        // INSTANTIATE_SOFTMAX(int64_t);

    }
}
