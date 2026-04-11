#include "slice.h"

namespace simplenet {
    namespace utils {

        // TODO
        SliceReturn computing_slice_parameters(std::vector<int> shape, std::vector<int> strides, ull offset, ull dim, Slice slice) {
            return SliceReturn(shape, strides, offset); // TODO: remove this line - boilerplace
            // TODO: complete this as this can be used by the slice function inside the Tensor class to compute the slice

        }



    }
}
