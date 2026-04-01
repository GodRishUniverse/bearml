#include "element_wise_kernels.cuh"
#include <cstdint>
#include <type_traits>


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
            OP_Code op_code
        ){

        for (size_t thread_idx = blockIdx.x*blockDim.x + threadIdx.x; thread_idx < res_flat_shape; thread_idx += blockDim.x * gridDim.x) {
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
                    case OP_Code::OP_MAX: {
                        result = (a[offA] > b[offB]) ? a[offA] : b[offB];
                        break;
                    }
                    case OP_Code::OP_MIN: {
                        result = (a[offA] < b[offB]) ? a[offA] : b[offB];
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
            for (size_t thread_idx = blockIdx.x*blockDim.x + threadIdx.x; thread_idx < n; thread_idx += blockDim.x * gridDim.x) {
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
                    case OP_Code::OP_MAX:{
                        result = (a[thread_idx] > b[thread_idx]) ? a[thread_idx] : b[thread_idx];
                        break;
                    }
                    case OP_Code::OP_MIN:{
                        result = (a[thread_idx] < b[thread_idx]) ? a[thread_idx] : b[thread_idx];
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


        // template <typename T>
        // __global__
        // void element_wise_broadcast_with_constant(
        //     const size_t* __restrict__ strides_a,
        //     const size_t* __restrict__ res_shape,
        //     size_t res_flat_shape,
        //     size_t outShapeSize,
        //     const T* __restrict__ a,
        //     const T* __restrict__ b,
        //     T* res,
        //     OP_Code op_code,
        //     LHS_RHS_Code lhs_rhs_code
        // ){
        //     size_t thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

        //     if (thread_idx< res_flat_shape){
        //         size_t tmp = thread_idx;
        //         size_t offA = 0;

        //         for (int d = outShapeSize - 1; d >= 0; --d) {
        //             size_t coord  = tmp % res_shape[d];
        //             tmp /= res_shape[d];
        //             offA += coord * strides_a[d];
        //         }

        //         T result;
        //         switch (op_code) {
        //             case OP_Code::OP_ADD: {
        //                 result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b + a[offA] ) : (a[offA] + *b);
        //                 break;
        //             }
        //             case OP_Code::OP_SUB:{
        //                 result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b - a[offA] ) : (a[offA] - *b);
        //                 break;
        //             }
        //             case OP_Code::OP_MUL: {
        //                 result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b * a[offA] ) : (a[offA] * *b);
        //                 break;
        //             }
        //             case OP_Code::OP_DIV: {
        //                 result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b / a[offA] ) : (a[offA] / *b);
        //                 break;
        //             }
        //             case OP_Code::OP_MAX: {
        //                 result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? ((*b > a[offA]) ? *b : a[offA] ) : ((a[offA] > *b) ? a[offA] : *b);
        //                 break;
        //             }
        //             case OP_Code::OP_MIN: {
        //                 result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? ((*b < a[offA]) ? *b : a[offA] ) : ((a[offA] < *b) ? a[offA] : *b);
        //                 break;
        //             }

        //             // default operation -> since we cannot throw an error inside the kernel
        //             default:
        //                 result = static_cast<T>(NAN);
        //                 break;
        //         }
        //         res[thread_idx] = result;
        //     }
        // }

        // // ---------------------------------- Template specification for floats ----------------------------------
        // // bfloat16
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<__nv_bfloat16 >(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const __nv_bfloat16*  a, const __nv_bfloat16*  b, __nv_bfloat16* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);

        // // float16
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<__half >(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const __half*  a, const __half*  b, __half* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);

        // // float32
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<float >(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const float*  a, const float*  b, float* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);

        // // float64
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<double >(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const double*  a, const double*  b, double* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);

        // // ---------------------------------- Template specification for ints ----------------------------------
        // // int8
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<int8_t>(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const int8_t*  a, const int8_t*  b, int8_t* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);
        // // int16
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<int16_t>(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const int16_t*  a, const int16_t*  b, int16_t* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);
        // // int32
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<int32_t>(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const int32_t*  a, const int32_t*  b, int32_t* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);
        // // int64
        // template __global__ void simplenet::cuda::element_wise_broadcast_with_constant<int64_t>(const size_t*  strides_a, const size_t*  res_shape, size_t res_flat_shape, size_t outShapeSize, const int64_t*  a, const int64_t*  b, int64_t* res, OP_Code op_code, LHS_RHS_Code lhs_rhs_code);

        //TODO : fix kernel for when the datasize is larger than the number of threads (this wont compute it) AS THREAD_IDX WILL NEVER REACH N


        template <typename T>
        __global__
        void element_wise_contiguous_with_constant(
            const T* __restrict__ a,
            const T* __restrict__ b,
            T* res,
            size_t n,
            OP_Code op_code,
            LHS_RHS_Code lhs_rhs_code
        ){
            for (size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < n; thread_idx += blockDim.x * gridDim.x) {
                T result;
                switch (op_code) {

                    case OP_Code::OP_ADD: {
                        result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b + a[thread_idx] ) : (a[thread_idx] + *b);
                        break;
                    }
                    case OP_Code::OP_SUB:{
                        result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b - a[thread_idx] ) : (a[thread_idx] - *b);
                        break;
                    }
                    case OP_Code::OP_MUL: {
                        result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b * a[thread_idx] ) : (a[thread_idx] * *b);
                        break;
                    }
                    case OP_Code::OP_DIV: {
                        result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? (*b / a[thread_idx] ) : (a[thread_idx] / *b);
                        break;
                    }
                    case OP_Code::OP_MAX: {
                        result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? ((*b > a[thread_idx]) ? *b : a[thread_idx] ) : ((a[thread_idx] > *b) ? a[thread_idx] : *b);
                        break;
                    }
                    case OP_Code::OP_MIN: {
                        result = (lhs_rhs_code == LHS_RHS_Code::OP_LHS) ? ((*b < a[thread_idx]) ? *b : a[thread_idx] ) : ((a[thread_idx] < *b) ? a[thread_idx] : *b);
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



        template <typename T>
        __global__
        void element_wise_unary(
            const T* __restrict__ a,
            T* res,
            size_t n,
            OP_Code op_code
        ){
            for (size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
                 thread_idx < n;
                 thread_idx += blockDim.x * gridDim.x) {
                T result;
                switch (op_code) {
                    case OP_Code::OP_EXP: {
                        if constexpr (std::is_same_v<T, double>) {
                            result = ::exp(a[thread_idx]);
                        } else {
                            result = static_cast<T>(::expf(static_cast<float>(a[thread_idx])));
                        }
                        break;
                    }
                    case OP_Code::OP_LOG: {
                        if constexpr (std::is_same_v<T, double>) {
                            result = ::log(a[thread_idx]);
                        } else {
                            result = static_cast<T>(::logf(static_cast<float>(a[thread_idx])));
                        }
                        break;
                    }

                    case OP_Code::OP_ABS: {
                        if constexpr (std::is_same_v<T, double>) {
                            result = ::abs(a[thread_idx]);
                        } else {
                            result = static_cast<T>(::logf(static_cast<float>(a[thread_idx])));
                        }
                        break;
                    }
                    case OP_Code::OP_SQRT: {
                        if constexpr (std::is_same_v<T, double>) {
                            result = ::sqrt(a[thread_idx]);
                        } else {
                            result = static_cast<T>(::sqrtf(static_cast<float>(a[thread_idx])));
                        }
                        break;
                    }
                    // default operation -> since we cannot throw an error inside the kernel
                    default:
                        result = T{};
                        break;
                }
                res[thread_idx] = result;
            }
        }


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
            cudaStream_t stream
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
            dim3 grid = get_blocks(res_flat_shape, THREAD_COUNT);
            // ((res_flat_shape + THREAD_COUNT - 1) / THREAD_COUNT);

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
            cudaStream_t stream
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

            // Configuring kernel launch - this is from the  cuda convetions - although I prefer a different naming scheme
            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(n, THREAD_COUNT); // Number of blocks

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

        template <typename T>
        void launch_elementwise_contiguous_with_constant(
            const T* d_a,
            const T b,
            T* d_out,
            const std::vector<int>& res_shape, // same as d_a shape cause contiguous
            OP_Code op_code,
            LHS_RHS_Code lhs_rhs_code,
            cudaStream_t stream
        ) {

            bool own_stream = (stream == nullptr);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            T* d_b;
            CUDA_CHECK(cudaMallocAsync(&d_b, sizeof(T), stream));
            CUDA_CHECK(cudaMemcpyAsync(d_b, &b, sizeof(T), cudaMemcpyHostToDevice, stream));

            // Computing the flat shape of the result/a/b tensor
            size_t n = 1;
            for (size_t d = 0; d < res_shape.size(); ++d) {
                n *= res_shape[d];
            }

            // Configuring kernel launch - this is from the  cuda convetions - although I prefer a different naming scheme
            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(n, THREAD_COUNT); // Number of blocks

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            element_wise_contiguous_with_constant<T>
                <<<grid, block, 0, stream>>>(
                    d_a,
                    d_b,
                    d_out,
                    n,
                    op_code,
                    lhs_rhs_code
            );

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            CUDA_CHECK(cudaFreeAsync(d_b, stream));

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }


        template <typename T>
        void launch_elementwise_unary(
            const T* d_a,
            T* d_out,
            const std::vector<int>& res_shape, // same as d_a and d_b shape cause contiguous
            OP_Code op_code,
            cudaStream_t stream
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

            // Configuring kernel launch - this is from the  cuda convetions - although I prefer a different naming scheme
            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(n, THREAD_COUNT); // Number of blocks

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            element_wise_unary<T>
                <<<grid, block, 0, stream>>>(
                    d_a,
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


        // floats
        INSTANTIATE_ELEMENT_WISE(__nv_bfloat16);
        INSTANTIATE_ELEMENT_WISE(__half);
        INSTANTIATE_ELEMENT_WISE(float);
        INSTANTIATE_ELEMENT_WISE(double);

        // ints
        INSTANTIATE_ELEMENT_WISE(int8_t);
        INSTANTIATE_ELEMENT_WISE(int16_t);
        INSTANTIATE_ELEMENT_WISE(int32_t);
        INSTANTIATE_ELEMENT_WISE(int64_t);



        template <typename T>
        __global__
        void sign_kernel(
            const T* __restrict__ a,
            T* res,
            size_t n
        ) {
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < n;
            idx += blockDim.x * gridDim.x) {
                T val = a[idx];
                res[idx] = (val > T(0)) ? T(1) : ((val < T(0)) ? T(-1) : T(0));
            }
        }

        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void simplenet::cuda::sign_kernel<__nv_bfloat16 >(const __nv_bfloat16*, __nv_bfloat16*, size_t);

        // float16
        template __global__ void simplenet::cuda::sign_kernel<__half >(const __half*, __half*, size_t);

        // float32
        template __global__ void simplenet::cuda::sign_kernel<float>(const float*, float*, size_t);

        // float64
        template __global__ void simplenet::cuda::sign_kernel<double>(const double*, double*, size_t);

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void simplenet::cuda::sign_kernel<int8_t >(const int8_t*, int8_t*, size_t);
        // int16
        template __global__ void simplenet::cuda::sign_kernel<int16_t >(const int16_t*, int16_t*, size_t);
        // int32
        template __global__ void simplenet::cuda::sign_kernel<int32_t>(const int32_t*, int32_t*, size_t);
        // int64
        template __global__ void simplenet::cuda::sign_kernel<int64_t>(const int64_t*, int64_t*, size_t);


        template <typename T>
        void launch_sign_contiguous(
            const T* d_a,
            T* d_out,
            const std::vector<int>& res_shape, // same as d_a and d_b shape cause contiguous
            cudaStream_t stream
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

            // Configuring kernel launch - this is from the  cuda convetions - although I prefer a different naming scheme
            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(n, THREAD_COUNT); // Number of blocks

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            sign_kernel<T>
                <<<grid, block, 0, stream>>>(
                    d_a,
                    d_out,
                    n            );

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }

        // Float types
        template void launch_sign_contiguous<float>(const float*, float*, const std::vector<int>&, cudaStream_t);
        template void launch_sign_contiguous<double>(const double*, double*, const std::vector<int>&, cudaStream_t);
        template void launch_sign_contiguous<__half>(const __half*, __half*, const std::vector<int>&, cudaStream_t);
        template void launch_sign_contiguous<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, const std::vector<int>&, cudaStream_t);

        // Int Types
        template void launch_sign_contiguous<int8_t>(const int8_t*, int8_t*, const std::vector<int>&, cudaStream_t);
        template void launch_sign_contiguous<int16_t>(const int16_t*, int16_t*, const std::vector<int>&, cudaStream_t);
        template void launch_sign_contiguous<int32_t>(const int32_t*, int32_t*, const std::vector<int>&, cudaStream_t);
        template void launch_sign_contiguous<int64_t>(const int64_t*, int64_t*, const std::vector<int>&, cudaStream_t);
    }
}
