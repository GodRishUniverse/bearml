#include "reductions.cuh"


namespace simplenet{
    namespace cuda {

        // TODO: implement atomicMul - can we template this?
        // Use atomicCAS - which is atomic compare-and-swap
        template<typename T>
        __device__ T atomicMul(T* address, T val)
        {
          // TODO: IMPLEMENT this - https://stackoverflow.com/questions/43354798/atomic-multiplication-and-division
          // The idea is here to keep tracking the old value and updating it atomically and also the address
          int* address_as_int = (int*)address;
          int old = *address_as_int, assumed; // TODO: what is this syntax?
          do {
            assumed = old;
            // READ device/gpu function documentation
            old = atomicCAS(address_as_int, assumed, __float_as_int(val * __float_as_int(assumed)));
          } while (assumed != old);
          return __int_as_float(old);
        }


        template<typename T>
        __global__ void prod_acc_kernel(const T* d_data,T* d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                int64_t outer = idx / offset_old; // gets the outer index from the old shape
                int64_t actual_offset     = idx % offset_new_shape; // mods to only get the inner index of the new_shape - actual offset
                int64_t out_idx = outer * offset_new_shape + actual_offset; // flat data representation with the outer "batch" index and the actual offset
                atomicMul<T>(&d_out[out_idx], d_data[idx]);
            }
        }

        template<typename T>
        void launch_prod_acc_kernel(T *d_data, T *d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size, cudaStream_t stream) {
            bool own_stream = (stream == nullptr);
            if (own_stream) {
                CUDA_CHECK(cudaStreamCreate(&stream));
            }

            dim3 block(THREAD_COUNT); // Threads per block
            dim3 grid = get_blocks(size, THREAD_COUNT); // Number of blocks
            prod_acc_kernel<T><<<grid, block, 0, stream>>>(d_data, d_out, offset_new_shape, offset_old, size);

            CUDA_CHECK(cudaGetLastError()); // this checks for Launch errors

            if (own_stream) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }
        // floats
        INSTANTIATE_PROD_ACC(__nv_bfloat16);
        INSTANTIATE_PROD_ACC(__half);
        INSTANTIATE_PROD_ACC(float);
        INSTANTIATE_PROD_ACC(double);

        // ints
        INSTANTIATE_PROD_ACC(int8_t);
        INSTANTIATE_PROD_ACC(int16_t);
        INSTANTIATE_PROD_ACC(int32_t);
        INSTANTIATE_PROD_ACC(int64_t);

    }
}
