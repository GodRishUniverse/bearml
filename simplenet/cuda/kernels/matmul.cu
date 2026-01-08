// https://siboehm.com/articles/22/CUDA-MMM <- link to go over
#include "../includes/helper.h"

namespace simplenet {
    namespace cuda {
        //TODO: write broadcasting done on the gpu matmul code

        template <typename T>
        __global__
        void gemm_kernel_broadcast(
            const int* __restrict__ batch_shape,
            int batch_shape_size,
            const int64_t* __restrict__ strides_a,
            const int64_t* __restrict__ strides_b,
            const int a_rows,
            const int a_cols,
            const int b_cols,
            int64_t total_batch_size,
            const T* __restrict__ a,
            const T* __restrict__ b,
            T* c
        ){
            // each thread computes one element of the output
            int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total_elements = total_batch_size * a_rows * b_cols;

            if (global_idx >= total_elements) return; // stop computation after the total elements

            // flattened is right now batch*row*col and we get [batch_index][row_index*col_index]
            // batch index is given by global_idx / c_row*c_col
            // and the column of this representation is given by global_idx % c_row*c_col
            // now we have [batch_index][row_index*col_index]
            // To get the row and column indices
            //  -> divide the column obtained above by b_cols to get the row index
            //  -> modulo the column obtained above by b_cols to get the column index
            int c_mat_size = a_rows*b_cols;  // a_rows * b_cols is c matrix size
            int64_t batch_index = global_idx / c_mat_size;
            // division gives us the row value and the column is given by modulus
            int64_t row = (global_idx % c_mat_size) / b_cols;
            int64_t col = (global_idx % c_mat_size) % b_cols;
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

            for (int k =0; k < a_cols; k++) {
                sum += a[offset_a+ row*a_cols+ k] * b[offset_b + k * b_cols + col];
                // offsets basically skip to the correct batch for the computation for the broadcasted dot product
            }
            c[global_idx] = sum;
        }


        // naive kernel right now
        template <typename T>
        __global__
        void gemm_kernel_contiguous(
            int batchsize,
            int i,
            int j,
            int k,
            T* __restrict__ a,
            T* __restrict__ b,
            T* c
        ){
            int column = blockIdx.x*blockDim.x + threadIdx.x;
            int row = blockIdx.y*blockDim.y +threadIdx.y;
            int batchId = blockIdx.z *blockDim.z + threadIdx.z;

            if (batchId < batchsize &&  row<i && column<k){
                T sum {}; // brace initialization
                for (int mid = 0; mid<j; mid++){
                    sum+= a[batchId*i*j+row*j+mid]*b[batchId*j*k+mid*k+column];
                }
                c[batchId*i*k + row*k + column] = sum;
            }
        }

        //TODO: write host code here



    }
}
