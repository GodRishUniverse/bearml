#include "../includes/helper.h"
#include "../includes/ops.h"
#include <cmath>
#include <cstddef>
#include <cstdint>


// __restrict__ keyword usage: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
namespace simplenet {
    namespace cuda {
        // restrict is an optimization measure that basically tells the compiler that these are separate data

        //TODO : fix kernel for when the datasize is larger than the number of threads (this wont compute it) AS THREAD_IDX WILL NEVER REACH N
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
            OP_Code op_code
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
                    case OP_Code::OP_ADD: {
                        result = a[offA]+b[offB];
                        break;
                    }
                    case OP_Code::OP_SUB:{
                        result = a[offA]-b[offB];
                        break;
                    }
                    case OP_Code::OP_MUL: {
                        result = a[offA]*b[offB];
                        break;
                    }
                    case OP_Code::OP_DIV: {
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



        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void simplenet::cuda::element_wise_broadcast<__nv_bfloat16 >(const size_t*, const size_t*,  const size_t*, size_t, size_t, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, OP_Code);

        // float16
        template __global__ void simplenet::cuda::element_wise_broadcast<__half>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const __half*, const __half*, __half*, OP_Code);

        // float32
        template __global__ void simplenet::cuda::element_wise_broadcast<float>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const float*, const float*, float*, OP_Code);

        // float64
        template __global__ void simplenet::cuda::element_wise_broadcast<double>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const double*, const double*,  double*, OP_Code);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void simplenet::cuda::element_wise_broadcast<int8_t>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const int8_t*, const int8_t*, int8_t*, OP_Code);
        // int16
        template __global__ void simplenet::cuda::element_wise_broadcast<int16_t>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const int16_t*, const int16_t*, int16_t*, OP_Code);
        // int32
        template __global__ void simplenet::cuda::element_wise_broadcast<int32_t>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const int32_t*, const int32_t*, int32_t*, OP_Code);
        // int64
        template __global__ void simplenet::cuda::element_wise_broadcast<int64_t>(const size_t*, const size_t*,  const size_t*, size_t, size_t, const int64_t*, const int64_t*, int64_t*, OP_Code);

        //TODO : fix kernel for when the datasize is larger than the number of threads (this wont compute it) AS THREAD_IDX WILL NEVER REACH N
        // CASE: NO BROADCASTING NEEDED
        template <typename T>
        __global__
        void element_wise_contiguous(
            const T* __restrict__ a,
            const T* __restrict__ b,
            T* res,
            size_t n,
            OP_Code op_code
        ){
            size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (thread_idx < n) {
                T result;
                switch (op_code) {
                    case OP_Code::OP_ADD: {
                        result = a[thread_idx]+b[thread_idx];
                        break;
                    }
                    case OP_Code::OP_SUB:{
                        result = a[thread_idx]-b[thread_idx];
                        break;
                    }
                    case OP_Code::OP_MUL:{
                        result = a[thread_idx]*b[thread_idx];
                        break;
                    }
                    case OP_Code::OP_DIV:{
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


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void simplenet::cuda::element_wise_contiguous<__nv_bfloat16 >(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t, OP_Code);

        // float16
        template __global__ void simplenet::cuda::element_wise_contiguous<__half >(const __half*, const __half*, __half*, size_t, OP_Code);

        // float32
        template __global__ void simplenet::cuda::element_wise_contiguous<float>(const float*, const float*, float*, size_t, OP_Code);

        // float64
        template __global__ void simplenet::cuda::element_wise_contiguous<double >(const double*, const double*, double*, size_t, OP_Code);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void simplenet::cuda::element_wise_contiguous<int8_t >(const int8_t*, const int8_t*, int8_t*, size_t, OP_Code);
        // int16
        template __global__ void simplenet::cuda::element_wise_contiguous<int16_t >(const int16_t*, const int16_t*, int16_t*, size_t, OP_Code);
        // int32
        template __global__ void simplenet::cuda::element_wise_contiguous<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t, OP_Code);
        // int64
        template __global__ void simplenet::cuda::element_wise_contiguous<int64_t >(const int64_t*, const int64_t*, int64_t*, size_t, OP_Code);



        // ---------------------------------------------- LAUNCHING CODE ----------------------------------------------

        // CUDA streams ensure that the operations occur sequentially -> we want [Allocation -> CopyToDevice]->[Kernel]->[Free]


        // TODO: calling code must have the device check for the data
        //  - right now the Launch code -> d_a, d_b and d_out already on device as the variable name implies
        template <typename T>
        void launch_elementwise_broadcast(
            const T* d_a,
            const T* d_b,
            T* d_out,
            const std::vector<int>& strides_a,
            const std::vector<int>& strides_b,
            const std::vector<int>& res_shape,
            OP_Code op_code,
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
            void* d_buffer = nullptr;

            // Using async allocation
            size_t total_size = (strides_a.size() + strides_b.size() + res_shape.size()) * sizeof(size_t);
            CUDA_CHECK(cudaMallocAsync(&d_buffer, total_size, stream));

            // pointer arithmetic to get the buffers on the device
            size_t* d_strides_a = static_cast<size_t*>(d_buffer);
            size_t* d_strides_b = d_strides_a + strides_a.size();
            size_t* d_res_shape = d_strides_b + strides_b.size();

            // packing the data on host side for one memcpy
            std::vector<size_t> host_buffer;
            host_buffer.reserve(strides_a.size() + strides_b.size() + res_shape.size());
            host_buffer.insert(host_buffer.end(), strides_a.begin(), strides_a.end());
            host_buffer.insert(host_buffer.end(), strides_b.begin(), strides_b.end());
            host_buffer.insert(host_buffer.end(), res_shape.begin(), res_shape.end());

            // copying the data from host → device
            // Using Async copies
            CUDA_CHECK(
                cudaMemcpyAsync(
                    d_buffer,
                    host_buffer.data(),
                    total_size,
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
            CUDA_CHECK(cudaFreeAsync(d_buffer, stream));

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }

        // TODO: calling code must have the device check for the data
        // right now the Launch code -> d_a, d_b and d_out already on device as the variable name implies
        template <typename T>
        void launch_elementwise_contiguous(
            const T* d_a,
            const T* d_b,
            T* d_out,
            const std::vector<int>& res_shape, // same as d_a and d_b shape cause contiguous
            OP_Code op_code,
            cudaStream_t stream = nullptr
        ) {

            bool own_stream = (stream == nullptr);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            // Computing the flat shape of the result/a/b tensor
            size_t n = 1;
            for (size_t d = 0; d < res_shape.size(); ++d) {
                n *= res_shape[d];
            }

            // Configuring kernel launch
            dim3 block(THREAD_COUNT);
            dim3 grid((n + THREAD_COUNT - 1) / THREAD_COUNT);

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            element_wise_contiguous<T>
                <<<grid, block, 0, stream>>>(
                    d_a,
                    d_b,
                    d_out,
                    n,
                    op_code
            );

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }
    }
}
