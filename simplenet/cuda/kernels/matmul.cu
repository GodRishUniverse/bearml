// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include "matmul.cuh"
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
            int thread_id = threadIdx.x;

            int column = blockIdx.x* BLOCK_SIZE + (thread_id / BLOCK_SIZE); // rows access
            int row = blockIdx.y* BLOCK_SIZE + (thread_id % BLOCK_SIZE); // column access
            int batchId = blockIdx.z;

            if (batchId < batchsize &&  row<M && column<N){
                T sum {}; // brace initialization
                for (int mid = 0; mid<K; ++mid){
                    sum+= a[batchId*M*K + row*K + mid]*b[batchId*K*N + mid*N + column];
                }
                c[batchId*M*N + row*N + column] = alpha*sum + beta*c[batchId*M*N + row*N + column];
            }
        }

        //TODO: write host code here



    }
}
