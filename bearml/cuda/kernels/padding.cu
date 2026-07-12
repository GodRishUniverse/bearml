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
            int out_rows,
            int out_cols,
            int padding,
            int r_start,
            int r_end,
            int c_start,
            int c_end
        ) {
                const int in_plane   = in_rows * in_cols;
                const int out_plane  = out_rows * out_cols;
                const int copy_rows  = r_end - r_start;
                const int copy_cols  = c_end - c_start;
                const int64_t total_copy = (int64_t)batch_size * copy_rows * copy_cols;

                // Only iterate the region where input and (possibly cropped) output overlap.
                // Walking the full input range and offsetting by `padding` only stays in-bounds
                // for padding >= 0; for padding < 0 (cropping) it writes off both ends of d_out.
                for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_copy; idx += (int64_t) blockDim.x * gridDim.x) {
                    int batch = idx / (copy_rows * copy_cols);
                    int rem   = idx - batch * (copy_rows * copy_cols);

                    int row   = r_start + rem / copy_cols;
                    int col   = c_start + rem - (row - r_start) * copy_cols;

                    int64_t in_idx  = (int64_t)batch * in_plane + row * in_cols + col;
                    int64_t out_idx = (int64_t)batch * out_plane + (int64_t)(row + padding) * out_cols + (col + padding);
                    d_out[out_idx] = d_in[in_idx];
                }
        }


        template<typename T>
        void launch_padd_with_constant(
            T* d_in,
            T* d_out,
            int batch_size,
            int in_rows,
            int in_cols,
            int out_rows,
            int out_cols,
            int padding,
            int r_start,
            int r_end,
            int c_start,
            int c_end,
            T constant,
            cudaStream_t stream
        ) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            // we fill first with constant value (use our fill kernel)
            std::vector<int> res_shape = {batch_size, out_rows, out_cols};
            launch_fill<T>(d_out, constant, res_shape, stream);

            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks((size_t)batch_size * (r_end - r_start) * (c_end - c_start), THREAD_COUNT); // Number of blocks
            padd_with_constant_kernel<T><<<grid, block, 0, stream>>>(d_in, d_out, batch_size, in_rows, in_cols, out_rows, out_cols, padding, r_start, r_end, c_start, c_end);

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
