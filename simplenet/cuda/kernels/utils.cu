#include "utils.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

namespace simplenet {
    namespace cuda {
        namespace utils {
            // Kernel to cast elements from one type to another
            template <typename InputT, typename OutputT>
            __global__ void castTypeKernel(const InputT* d_input, OutputT* d_output, int N) {
                for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
                    d_output[idx] = static_cast<OutputT>(d_input[idx]);
                }
            }

            template <typename InputT, typename OutputT>
            void launch_dtype_change(InputT *d_data, OutputT *d_out, int64_t size, cudaStream_t stream ) {
                bool own_stream = (stream == nullptr);
                if (own_stream) {
                    CUDA_CHECK(cudaStreamCreate(&stream));
                }

                dim3 block(THREAD_COUNT); // Threads per block
                // total size of the grid is outerDim
                dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks

                castTypeKernel<InputT, OutputT><<<grid, block, 0, stream>>>(d_data, d_out, size);

                CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

                if (own_stream) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaStreamDestroy(stream));
                }
            }

            // Stride-walking gather: writes a row-major contiguous copy of a strided source.
            // Used by Tensor::contiguous() when the source is a view (transpose / permute / slice / broadcast),
            // so the source's storage may have gaps, repeats, or a reversed walk order vs row-major.
            template <typename T>
            __global__ void contiguous_gather_kernel(
                const T* __restrict__ d_src, T* __restrict__ d_dst, size_t src_offset,
                const size_t* __restrict__ d_shape, const size_t* __restrict__ d_strides,
                size_t nd, size_t total
            ) {
                // grid-stride loop: each thread keeps grabbing a fresh output index until we've covered all `total` elements
                for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                     i < total;
                     i += (size_t)blockDim.x * gridDim.x) {
                    // i is the flat row-major index into the DESTINATION (we're building it dense and row-major).
                    // To copy the right value we need to figure out which (i0, i1, ..., i_{nd-1}) coord that is,
                    // then index into the SOURCE using the source's strides (which may be anything).
                    size_t tmp = i;          // running quotient — every iteration of the inner loop peels off one axis from it
                    size_t src = src_offset; // running flat index into source storage; starts at the view's offset (slice/broadcast views can carry a nonzero offset)

                    // walk axes innermost -> outermost because in ROW-MAJOR the innermost (last) dim is the
                    // fastest-changing one in i, so it modulo's out cleanly first. Then the next outer dim, and so on.
                    for (int d = (int)nd - 1; d >= 0; --d) {
                        size_t coord = tmp % d_shape[d]; // coord along axis d (e.g. column when d == nd-1, then row when d == nd-2, ...)
                        tmp /= d_shape[d];               // strip that axis off so the next iteration sees only the more-outer dims
                        src += coord * d_strides[d];     // step `coord` times in source storage along axis d
                                                         // -> source stride is 0 for broadcast axes (we re-read the same element),
                                                         // -> source stride for a transposed dim is the OTHER dim's row-major stride,
                                                         // -> for an arbitrary permute it's whichever original axis was permuted into d.
                    }

                    d_dst[i] = d_src[src]; // dst is dense row-major, so its flat index IS i; src is wherever the source's stride pattern landed
                }
            }

            template <typename T>
            void launch_contiguous_gather(
                const T* d_src, T* d_dst, size_t src_offset,
                const size_t* h_shape, const size_t* h_strides, size_t nd, size_t total,
                cudaStream_t stream
            ) {
                bool own_stream = (stream == nullptr);
                if (own_stream) {
                    CUDA_CHECK(cudaStreamCreate(&stream));
                }

                size_t md_bytes = nd * sizeof(size_t);
                size_t *d_shape = nullptr, *d_strides = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_shape,   md_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_strides, md_bytes, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_shape,   h_shape,   md_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_strides, h_strides, md_bytes, cudaMemcpyHostToDevice, stream));

                dim3 block(THREAD_COUNT);
                dim3 grid = get_blocks(total, THREAD_COUNT);
                contiguous_gather_kernel<T><<<grid, block, 0, stream>>>(
                    d_src, d_dst, src_offset, d_shape, d_strides, nd, total);

                CUDA_CHECK(cudaGetLastError());

                CUDA_CHECK(cudaFreeAsync(d_shape,   stream));
                CUDA_CHECK(cudaFreeAsync(d_strides, stream));

                if (own_stream) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaStreamDestroy(stream));
                }
            }

            INSTANTIATE_SINGLE_TYPE_UTILS(float)
            INSTANTIATE_SINGLE_TYPE_UTILS(double)
            INSTANTIATE_SINGLE_TYPE_UTILS(__half)
            INSTANTIATE_SINGLE_TYPE_UTILS(__nv_bfloat16)
            INSTANTIATE_SINGLE_TYPE_UTILS(int8_t)
            INSTANTIATE_SINGLE_TYPE_UTILS(int16_t)
            INSTANTIATE_SINGLE_TYPE_UTILS(int32_t)
            INSTANTIATE_SINGLE_TYPE_UTILS(int64_t)

            // floats
            INSTANTIATE_DTYPE_CHANGE(double, float)
            INSTANTIATE_DTYPE_CHANGE(double, __half)
            INSTANTIATE_DTYPE_CHANGE(double, __nv_bfloat16)

            INSTANTIATE_DTYPE_CHANGE(float, __half)
            INSTANTIATE_DTYPE_CHANGE(float, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(float, double)

            INSTANTIATE_DTYPE_CHANGE(__half, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(__half, double)
            INSTANTIATE_DTYPE_CHANGE(__half, float)

            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, __half)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, double)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, float)

            // ints
            INSTANTIATE_DTYPE_CHANGE(int32_t, int64_t)
            INSTANTIATE_DTYPE_CHANGE(int32_t, int16_t)
            INSTANTIATE_DTYPE_CHANGE(int32_t, int8_t)

            INSTANTIATE_DTYPE_CHANGE(int64_t, int32_t)
            INSTANTIATE_DTYPE_CHANGE(int64_t, int16_t)
            INSTANTIATE_DTYPE_CHANGE(int64_t, int8_t)

            INSTANTIATE_DTYPE_CHANGE(int16_t, int32_t)
            INSTANTIATE_DTYPE_CHANGE(int16_t, int64_t)
            INSTANTIATE_DTYPE_CHANGE(int16_t, int8_t)

            INSTANTIATE_DTYPE_CHANGE(int8_t, int32_t)
            INSTANTIATE_DTYPE_CHANGE(int8_t, int64_t)
            INSTANTIATE_DTYPE_CHANGE(int8_t, int16_t)

            // ints and floats
            INSTANTIATE_DTYPE_CHANGE(int32_t, float)
            INSTANTIATE_DTYPE_CHANGE(int64_t, float)
            INSTANTIATE_DTYPE_CHANGE(int16_t, float)
            INSTANTIATE_DTYPE_CHANGE(int8_t, float)

            INSTANTIATE_DTYPE_CHANGE(float, int32_t)
            INSTANTIATE_DTYPE_CHANGE(float, int64_t)
            INSTANTIATE_DTYPE_CHANGE(float, int16_t)
            INSTANTIATE_DTYPE_CHANGE(float, int8_t)

            // ints and doubles
            INSTANTIATE_DTYPE_CHANGE(int32_t, double)
            INSTANTIATE_DTYPE_CHANGE(int64_t, double)
            INSTANTIATE_DTYPE_CHANGE(int16_t, double)
            INSTANTIATE_DTYPE_CHANGE(int8_t, double)

            INSTANTIATE_DTYPE_CHANGE(double, int32_t)
            INSTANTIATE_DTYPE_CHANGE(double, int64_t)
            INSTANTIATE_DTYPE_CHANGE(double, int16_t)
            INSTANTIATE_DTYPE_CHANGE(double, int8_t)

            // ints and __half
            INSTANTIATE_DTYPE_CHANGE(int32_t, __half)
            INSTANTIATE_DTYPE_CHANGE(int64_t, __half)
            INSTANTIATE_DTYPE_CHANGE(int16_t, __half)
            INSTANTIATE_DTYPE_CHANGE(int8_t, __half)

            INSTANTIATE_DTYPE_CHANGE(__half, int32_t)
            INSTANTIATE_DTYPE_CHANGE(__half, int64_t)
            INSTANTIATE_DTYPE_CHANGE(__half, int16_t)
            INSTANTIATE_DTYPE_CHANGE(__half, int8_t)

            // ints and __nv_bfloat16
            INSTANTIATE_DTYPE_CHANGE(int32_t, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(int64_t, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(int16_t, __nv_bfloat16)
            INSTANTIATE_DTYPE_CHANGE(int8_t, __nv_bfloat16)

            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int32_t)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int64_t)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int16_t)
            INSTANTIATE_DTYPE_CHANGE(__nv_bfloat16, int8_t)

        }
    }
}
