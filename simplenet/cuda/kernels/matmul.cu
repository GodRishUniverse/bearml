// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include "matmul.cuh"
#include <cstdint>
// gemm and why we need -> alpha*a*b + beta*c
// https://math.stackexchange.com/questions/1826305/significance-of-alpha-and-beta-with-regards-to-matrix-multiplication
//
namespace simplenet {
    namespace cuda {
        // TODO: this needs to fixed a lot more and tested
        // TODO: Optimize - using memory coalescing, warp level parallelism, and shared memory, and tiling - check blog above and also look more into this
        //TODO: write broadcasting done on the gpu matmul code
        //TODO : fix kernel for when the datasize is larger than the number of threads (this wont compute it)
        template <typename T>
        __global__
        void gemm_kernel_broadcast(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            T alpha, T beta,
            int64_t total_batch_size,
            const T* __restrict__ a,
            const T* __restrict__ b,
            T* c
        ){
            // each thread computes one element of the output
            int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total_elements = total_batch_size * M * N;

            if (global_idx >= total_elements) return; // stop computation after the total elements

            // flattened is right now batch*row*col and we get [batch_index][row_index*col_index]
            // batch index is given by global_idx / c_row*c_col
            // and the column of this representation is given by global_idx % c_row*c_col
            // now we have [batch_index][row_index*col_index]
            // To get the row and column indices
            //  -> divide the column obtained above by b_cols to get the row index
            //  -> modulo the column obtained above by b_cols to get the column index
            int c_mat_size = M*N;  // a_rows * b_cols is c matrix size
            int64_t batch_index = global_idx / c_mat_size;
            // division gives us the row value and the column is given by modulus
            int64_t row = (global_idx % c_mat_size) / N;
            int64_t col = (global_idx % c_mat_size) % N;
            // now we get [batch_index][row_index][col_index] as indices values

            int64_t tmp = batch_index;
            int64_t offset_a = 0, offset_b = 0;

            // calculate the offsets and the broadcasted coordinates of the original mat
            for (int d = batch_shape_size - 1; d >= 0; --d) {
                int coord = (batch_shape[d] > 0) ? (tmp % batch_shape[d]) : 0;
                tmp /= batch_shape[d];
                offset_a += coord * strides_a[d];
                offset_b += coord * strides_b[d];
            }

            // computing the dot product -> instead of using Eigen
            T sum = 0;

            for (int k =0; k < K; k++) {
                sum += a[offset_a+ row*K+ k] * b[offset_b + k * N + col];
                // offsets basically skip to the correct batch for the computation for the broadcasted dot product
            }
            c[global_idx] = alpha*sum+beta*c[global_idx]; // gemm
        }


        // coelsced kernel - need to change optimize further
        // optimize - look over ->  https://siboehm.com/articles/22/CUDA-MMM
        //TODO : fix kernel for when the datasize is larger than the number of threads (this wont compute it)

        template <typename T>
        __global__
        void gemm_kernel_contiguous(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            T alpha, T beta,
            T* __restrict__ a,
            T* __restrict__ b,
            T* c
        ){
            // int thread_id = threadIdx.x;

            int column = blockIdx.x* BLOCK_SIZE + threadIdx.x; // rows access
            int row = blockIdx.y* BLOCK_SIZE + threadIdx.y; // column access
            int batchId = blockIdx.z;

            if (batchId < batchsize &&  row<M && column<N){
                T sum {}; // brace initialization
                for (int mid = 0; mid<K; ++mid){
                    sum+= a[batchId*M*K + row*K + mid]*b[batchId*K*N + mid*N + column];
                }
                c[batchId*M*N + row*N + column] = alpha*sum + beta*c[batchId*M*N + row*N + column];
            }
        }


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<__nv_bfloat16>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            __nv_bfloat16 alpha, __nv_bfloat16 beta,
            int64_t total_batch_size,
            const __nv_bfloat16* __restrict__ a,
            const __nv_bfloat16* __restrict__ b,
            __nv_bfloat16* c
        );

        // float16
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<__half>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            __half alpha, __half beta,
            int64_t total_batch_size,
            const __half* __restrict__ a,
            const __half* __restrict__ b,
            __half* c
        );

        // float32
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<float>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            float alpha, float beta,
            int64_t total_batch_size,
            const float* __restrict__ a,
            const float* __restrict__ b,
            float* c
        );

        // float64
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<double>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            double alpha, double beta,
            int64_t total_batch_size,
            const double* __restrict__ a,
            const double* __restrict__ b,
            double* c
        );

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<int8_t>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            int8_t alpha, int8_t beta,
            int64_t total_batch_size,
            const int8_t* __restrict__ a,
            const int8_t* __restrict__ b,
            int8_t* c
        );
        // int16
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<int16_t>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            int16_t alpha, int16_t beta,
            int64_t total_batch_size,
            const int16_t* __restrict__ a,
            const int16_t* __restrict__ b,
            int16_t* c
        );
        // int32
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<int32_t>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            int32_t alpha, int32_t beta,
            int64_t total_batch_size,
            const int32_t* __restrict__ a,
            const int32_t* __restrict__ b,
            int32_t* c
        );
        // int64
        template __global__ void simplenet::cuda::gemm_kernel_broadcast<int64_t>(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int M,
            const int K,
            const int N,
            int64_t alpha, int64_t beta,
            int64_t total_batch_size,
            const int64_t* __restrict__ a,
            const int64_t* __restrict__ b,
            int64_t* c
        );


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<__nv_bfloat16>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            __nv_bfloat16 alpha, __nv_bfloat16 beta,
            __nv_bfloat16* __restrict__ a,
            __nv_bfloat16* __restrict__ b,
            __nv_bfloat16* c
        );

        // float16
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<__half>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            __half alpha, __half beta,
            __half* __restrict__ a,
            __half* __restrict__ b,
            __half* c
        );

        // float32
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<float>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            float alpha, float beta,
            float* __restrict__ a,
            float* __restrict__ b,
            float* c
        );

        // float64
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<double>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            double alpha, double beta,
            double* __restrict__ a,
            double* __restrict__ b,
            double* c
        );

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<int8_t>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            int8_t alpha, int8_t beta,
            int8_t* __restrict__ a,
            int8_t* __restrict__ b,
            int8_t* c
        );

        // int16
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<int16_t>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            int16_t alpha, int16_t beta,
            int16_t* __restrict__ a,
            int16_t* __restrict__ b,
            int16_t* c
        );
        // int32
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<int32_t>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            int32_t alpha, int32_t beta,
            int32_t* __restrict__ a,
            int32_t* __restrict__ b,
            int32_t* c
        );
        // int64
        template __global__ void simplenet::cuda::gemm_kernel_contiguous<int64_t>(
            int batchsize,
            const int M, // row_count
            const int K, // common_dim
            const int N, // column_count
            int64_t alpha, int64_t beta,
            int64_t* __restrict__ a,
            int64_t* __restrict__ b,
            int64_t* c
        );


        // test this out:
        template<typename T>
        void launch_gemm_broadcasted(
            const T* d_a,
            const T* d_b,
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
        ){
            bool own_stream = (stream == nullptr);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            // Copy data from host to device
            int64_t* d_strides_a = nullptr;
            int64_t* d_strides_b = nullptr;
            int* d_batch_shape = nullptr;
            CUDA_CHECK(cudaMalloc(&d_batch_shape, batch_shape_size*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_strides_a, strides_a->size() * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&d_strides_b, strides_b->size() * sizeof(int64_t)));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_shape, batch_shape->data(), batch_shape_size*sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_strides_a, strides_a->data(), strides_a->size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_strides_b, strides_b->data(), strides_b->size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));


            // Configuring kernel launch - this is from the  cuda convetions - although I prefer a different naming scheme
            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks( total_batch_size * m * n, THREAD_COUNT); // Number of blocks

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            gemm_kernel_broadcast<T><<<grid, block, 0, stream>>>(d_batch_shape, batch_shape_size,d_strides_a, d_strides_b,m , k, n, alpha, beta, total_batch_size, d_a, d_b, d_c);


            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            CUDA_CHECK(cudaFree(d_strides_a));
            CUDA_CHECK(cudaFree(d_strides_b));
            CUDA_CHECK(cudaFree(d_batch_shape));


            if (own_stream) {
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }


        // test this out
        template<typename T>
        void launch_gemm_contiguous(
            T* d_a,
            T* d_b,
            T* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            T alpha,
            T beta,
            cudaStream_t stream
        ) {
            bool own_stream = (stream == nullptr);
            // if we do need to create a stream then we create it here
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            // Configuring kernel launch - this is from the  cuda convetions - although I prefer a different naming scheme
            dim3 block(BLOCK_SIZE, BLOCK_SIZE); // Threads per block
            dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      batchsize);

            // launching the kernel - syntax kernel_name<<<grid, block, sharedMem, stream>>>(kernel_args);
            gemm_kernel_contiguous<T><<<grid, block, 0, stream>>>(batchsize, m,  k, n,  alpha, beta, d_a, d_b, d_c);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template
        void launch_gemm_broadcasted<__nv_bfloat16>(
            const __nv_bfloat16* d_a,
            const __nv_bfloat16* d_b,
            __nv_bfloat16* d_c,
            int m,
            int k,
            int n,
            __nv_bfloat16 alpha,
            __nv_bfloat16 beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );

        // float16
        template
        void launch_gemm_broadcasted<__half>(
            const __half* d_a,
            const __half* d_b,
            __half* d_c,
            int m,
            int k,
            int n,
            __half alpha,
            __half beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );

        // float32
        template
        void launch_gemm_broadcasted<float>(
            const float* d_a,
            const float* d_b,
            float* d_c,
            int m,
            int k,
            int n,
            float alpha,
            float beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );

        // float64
        template
        void launch_gemm_broadcasted<double>(
            const double* d_a,
            const double* d_b,
            double* d_c,
            int m,
            int k,
            int n,
            double alpha,
            double beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template
        void launch_gemm_broadcasted<int8_t>(
            const int8_t* d_a,
            const int8_t* d_b,
            int8_t* d_c,
            int m,
            int k,
            int n,
            int8_t alpha,
            int8_t beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );
        // int16
        template
        void launch_gemm_broadcasted<int16_t>(
            const int16_t* d_a,
            const int16_t* d_b,
            int16_t* d_c,
            int m,
            int k,
            int n,
            int16_t alpha,
            int16_t beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );
        // int32
        template
        void launch_gemm_broadcasted<int32_t>(
            const int32_t* d_a,
            const int32_t* d_b,
            int32_t* d_c,
            int m,
            int k,
            int n,
            int32_t alpha,
            int32_t beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );
        // int64
        template
        void launch_gemm_broadcasted<int64_t>(
            const int64_t* d_a,
            const int64_t* d_b,
            int64_t* d_c,
            int m,
            int k,
            int n,
            int64_t alpha,
            int64_t beta,
            std::vector<int>* batch_shape,
            int batch_shape_size,
            std::vector<int64_t>* strides_a,
            std::vector<int64_t>* strides_b,
            int64_t total_batch_size,
            cudaStream_t stream
        );


        // ---------------------------------- Template specification for floats ----------------------------------
        // bfloat16
        template
        void launch_gemm_contiguous<__nv_bfloat16>(
            __nv_bfloat16* d_a,
            __nv_bfloat16* d_b,
            __nv_bfloat16* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            __nv_bfloat16 alpha,
            __nv_bfloat16 beta,
            cudaStream_t stream
        );

        // float16
        template
        void launch_gemm_contiguous<__half>(
            __half* d_a,
            __half* d_b,
            __half* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            __half alpha,
            __half beta,
            cudaStream_t stream
        );

        // float32
        template
        void launch_gemm_contiguous<float>(
            float* d_a,
            float* d_b,
            float* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            float alpha,
            float beta,
            cudaStream_t stream
        );

        // float64
        template
        void launch_gemm_contiguous<double>(
            double* d_a,
            double* d_b,
            double* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            double alpha,
            double beta,
            cudaStream_t stream
        );

        // ---------------------------------- Template specification for ints ----------------------------------
        // int8
        template
        void launch_gemm_contiguous<int8_t>(
            int8_t* d_a,
            int8_t* d_b,
            int8_t* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            int8_t alpha,
            int8_t beta,
            cudaStream_t stream
        );
        // int16
        template
        void launch_gemm_contiguous<int16_t>(
            int16_t* d_a,
            int16_t* d_b,
            int16_t* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            int16_t alpha,
            int16_t beta,
            cudaStream_t stream
        );
        // int32
        template
        void launch_gemm_contiguous<int32_t>(
            int32_t* d_a,
            int32_t* d_b,
            int32_t* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            int32_t alpha,
            int32_t beta,
            cudaStream_t stream
        );
        // int64
        template
        void launch_gemm_contiguous<int64_t>(
            int64_t* d_a,
            int64_t* d_b,
            int64_t* d_c,
            int batchsize,
            int m,
            int k,
            int n,
            int64_t alpha,
            int64_t beta,
            cudaStream_t stream
        );


    }
}
