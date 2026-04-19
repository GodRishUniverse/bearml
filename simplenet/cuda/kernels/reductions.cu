#include "reductions.cuh"


namespace simplenet{
    namespace cuda {

        // utility function: bit_cast - converts between types of the same size - uses memcpy for the actual conversion
        template <typename AsType, typename FromType>
        __device__ AsType bit_cast(FromType val) {
            static_assert(sizeof(AsType) == sizeof(FromType), "size mismatch");
            AsType result;
            memcpy(&result, &val, sizeof(val));
            return result;
        }

        // atomicMul via atomicCAS (compare-and-swap loop)
        // Has to have CASTraits<T> defined
        template<typename T>
        __device__ T atomicMul(T* address, T val)
        {
          using U = typename simplenet::cuda::CASTraits<T>::int_type;

          U* address_as_u = reinterpret_cast<U*>(address); // reinterpret address as U* for atomicCAS
          U old = *address_as_u;
          U assumed_bits;

          do {
                assumed_bits = old;
                T assumed_val = bit_cast<T, U>(assumed_bits);
                T desired_val = assumed_val * val;
                U desired_bits = bit_cast<U, T>(desired_val);
                old = atomicCAS(address_as_u, assumed_bits, desired_bits);
          } while (assumed_bits != old);

          return bit_cast<T, U>(old);
        }

        // TODO: - think about int8_t specialization - so we apply bitwise operations - understand overlying architecture and operation to figure out the operation



        template<typename T>
        __global__ void prod_acc_kernel(const T* d_data,T* d_out, int64_t offset_new_shape, int64_t offset_old, int64_t size) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  idx < size; idx += blockDim.x * gridDim.x) {
                int64_t outer = idx / offset_old; // gets the outer index from the old shape
                int64_t actual_offset     = idx % offset_new_shape; // mods to only get the inner index of the new_shape - actual offset
                int64_t out_idx = outer * offset_new_shape + actual_offset; // flat data representation with the outer "batch" index and the actual offset
                atomicMul<T>(&d_out[out_idx], d_data[idx]);
            }
        }

        // test this
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
        // INSTANTIATE_PROD_ACC(int8_t);
        INSTANTIATE_PROD_ACC(int16_t);
        INSTANTIATE_PROD_ACC(int32_t);
        INSTANTIATE_PROD_ACC(int64_t);

    }
}
