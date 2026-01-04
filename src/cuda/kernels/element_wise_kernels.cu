#include "../includes/helper.h"
#include "../includes/ops.h"


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


        // CUDA streams ensure that the operations occur sequentially -> we want [Allocation -> CopyToDevice]->[Kernel]->[Free]
        // Launch code -> d_a, d_b and d_out already on device as the variable name implies
        template <typename T>
        void launch_elementwise_broadcast(
            const T* d_a,
            const T* d_b,
            T* d_out,
            const std::vector<int>& strides_a,
            const std::vector<int>& strides_b,
            const std::vector<int>& res_shape,
            uint16_t op_code,
            cudaStream_t stream = nullptr
        ) {

            bool own_stream = (stream == nullptr);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            // Computing the flat shape of the result tensor
            size_t res_flat_shape = 1;
            for (size_t d = 0; d < res_shape.size(); ++d) {
                res_flat_shape *= res_shape[d];
            }

            // allocating the device memory for shape/stride
            size_t* d_strides_a = nullptr;
            size_t* d_strides_b = nullptr;
            size_t* d_res_shape = nullptr;

            // Using async allocation
            CUDA_CHECK(cudaMallocAsync(&d_strides_a, strides_a.size() * sizeof(size_t), stream));
            CUDA_CHECK(cudaMallocAsync(&d_strides_b, strides_b.size() * sizeof(size_t), stream));
            CUDA_CHECK(cudaMallocAsync(&d_res_shape, res_shape.size() * sizeof(size_t), stream));


            // copying the data from host → device
            // Using Async copies
            CUDA_CHECK(
                cudaMemcpyAsync(
                    d_strides_a,
                    strides_a.data(),
                    strides_a.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            CUDA_CHECK(
                cudaMemcpyAsync(
                    d_strides_b,
                    strides_b.data(),
                    strides_b.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            CUDA_CHECK(
                cudaMemcpyAsync(
                    d_res_shape,
                    res_shape.data(),
                    res_shape.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            const size_t resShapeSize = res_shape.size();


            // Configuring kernel launch
            dim3 block(THREAD_COUNT);
            dim3 grid((res_flat_shape + THREAD_COUNT - 1) / THREAD_COUNT);

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            element_wise_broadcast<T>
                <<<grid, block, 0, stream>>>(
                    d_strides_a,
                    d_strides_b,
                    d_res_shape,
                    res_flat_shape,
                    resShapeSize,
                    d_a,
                    d_b,
                    d_out,
                    op_code
            );

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            // cleanup after kernel finishes
            CUDA_CHECK(cudaFreeAsync(d_strides_a, stream));
            CUDA_CHECK(cudaFreeAsync(d_strides_b, stream));
            CUDA_CHECK(cudaFreeAsync(d_res_shape, stream));

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }
    }
}
