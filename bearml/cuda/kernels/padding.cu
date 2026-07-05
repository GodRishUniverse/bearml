#include "padding.cuh"
#include "fill.cuh"
#include <cstddef>


namespace bearml {
    namespace cuda {

        template<typename T>
        __global__ void padd_with_constant_kernel(
            const T* __restrict__ d_in,
            T* __restrict__ d_out,
            int batch_size,
            int in_rows,
            int in_cols,
            int padding
        ) {
                const int out_rows  = in_rows + 2 * padding;
                const int out_cols  = in_cols + 2 * padding;
                const int in_plane  = in_rows * in_cols;
                const int out_plane = out_rows * out_cols;
                const int64_t total_in = (int64_t )batch_size * in_plane;

                for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_in; idx += (int64_t) blockDim.x * gridDim.x) {
                    int batch = idx / in_plane; // get the batch index of the current element
                    int rem   = idx - batch * in_plane; // same as idx % in_plane but more efficient - uses mod arithmetic

                    int row   = rem / in_cols;
                    int col   = rem - row * in_cols;

                    int64_t out_idx = (int64_t)batch * out_plane+ (int64_t)(row + padding) * out_cols + (col + padding);
                    d_out[out_idx] = d_in[idx]; // copy the value from the input to the output with padding
                }
        }


        template<typename T>
        void launch_padd_with_constant(
            T* d_in,
            T* d_out,
            int batch_size,
            int in_rows,
            int in_cols,
            int padding,
            T constant,
            cudaStream_t stream
        ) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            // we fill first with constant value (use our fill kernel)
            std::vector<int> res_shape = {batch_size, in_rows + 2 * padding, in_cols + 2 * padding};
            launch_fill<T>(d_out, constant, res_shape, stream);


            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks((size_t)batch_size * in_rows * in_cols, THREAD_COUNT); // Number of blocks
            padd_with_constant_kernel<T><<<grid, block, 0, stream>>>(d_in, d_out, batch_size, in_rows, in_cols, padding);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }



        // floats
        INSTANTIATE_PADDING(__nv_bfloat16);
        INSTANTIATE_PADDING(__half);
        INSTANTIATE_PADDING(float);
        INSTANTIATE_PADDING(double);

        // ints
        INSTANTIATE_PADDING(int8_t);
        INSTANTIATE_PADDING(int16_t);
        INSTANTIATE_PADDING(int32_t);
        INSTANTIATE_PADDING(int64_t);
    }
}
