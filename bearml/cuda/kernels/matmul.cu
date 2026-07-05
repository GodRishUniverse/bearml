// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include "matmul.cuh"
#include <cstdint>
// gemm and why we need -> alpha*a*b + beta*c
// https://math.stackexchange.com/questions/1826305/significance-of-alpha-and-beta-with-regards-to-matrix-multiplication
//
namespace bearml {
    namespace cuda {
        // TODO: this needs to fixed a lot more and tested
        // TODO: Optimize - using memory coalescing, warp level parallelism, and shared memory, and tiling - check blog above and also look more into this
        //TODO: write broadcasting done on the gpu matmul code
        // TODO: remove % sign as it is very slow
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
            for (int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
                 global_idx < total_batch_size * M * N;
                 global_idx += blockDim.x * gridDim.x) {

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
            // Layout-aware GEMM - so that when we have ColumnMajor layout, we can still use contiguous memory access
            int64_t row_aware_strde_a, int64_t col_aware_stride_a,
            int64_t row_aware_strde_b, int64_t col_aware_stride_b,
            T* __restrict__ a,
            T* __restrict__ b,
            T* c
        ){
            int column = blockIdx.x* BLOCK_SIZE + threadIdx.x; // rows access
            int row = blockIdx.y* BLOCK_SIZE + threadIdx.y; // column access
            int batchId = blockIdx.z;

            if (batchId < batchsize &&  row<M && column<N){
                int64_t a_batch_off = (int64_t) batchId * M * K;
                int64_t b_batch_off = (int64_t) batchId * K * N;
                int64_t c_batch_off = (int64_t) batchId * M * N;

                T sum {}; // brace initialization
                for (int mid = 0; mid<K; ++mid){
                    // basically, we're doing a dot product between row/column of a and b - but stride-aware so that col/row-major order is preserved (triton kernel was the inspiration)
                    sum+= a[a_batch_off + row*row_aware_strde_a + mid*col_aware_stride_a]*b[b_batch_off + mid*row_aware_strde_b + column*col_aware_stride_b];
                }
                c[c_batch_off + row*N + column] = alpha*sum + beta*c[c_batch_off + row*N + column];
            }
        }

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
            int64_t row_aware_strde_a, int64_t col_aware_stride_a,
            int64_t row_aware_strde_b, int64_t col_aware_stride_b,
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
            gemm_kernel_contiguous<T><<<grid, block, 0, stream>>>(batchsize, m,  k, n,  alpha, beta,
                row_aware_strde_a, col_aware_stride_a,
                row_aware_strde_b, col_aware_stride_b,
                d_a, d_b, d_c);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }

        }

        // ---------------------------------- Template specification for floats ----------------------------------
        INSTANTIATE_GEMM(__nv_bfloat16);
        INSTANTIATE_GEMM(__half);
        INSTANTIATE_GEMM(float);
        INSTANTIATE_GEMM(double);

        // ---------------------------------- Template specification for ints ----------------------------------
        INSTANTIATE_GEMM(int8_t);
        INSTANTIATE_GEMM(int16_t);
        INSTANTIATE_GEMM(int32_t);
        INSTANTIATE_GEMM(int64_t);
    }
}
