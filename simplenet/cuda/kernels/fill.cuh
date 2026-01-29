#include "../includes/cuda_helper.h"


namespace simplenet {
    namespace cuda {

        template <typename T>
        void launch_fill(
            T* d_data,
            T value,
            std::vector<int>& res_shape,
            cudaStream_t stream = nullptr
        );
    }
}
