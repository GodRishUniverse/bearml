#include "slice.h"

namespace simplenet {
    namespace utils {


        SliceReturn computing_slice_parameters(std::vector<int> shape, std::vector<int> strides, std::vector<utils::Slice> slices) {
            std::vector<std::pair<int, int>> offsets_and_end_per_dim;
            // we will have to traverse in reverse order to compute the offsets and end per dimension
            for (int i = slices.size() - 1; i >= 0; --i) {
                // for last dimension, offset is slice.start and end is slice.end (however, we also need to account for the step size)

            }
            // TODO: slice return is wrong (need to think about how to compute the offset and end per dimension)
            // TODO: One thing I learned is that for n-dim slicing to function offset is needed but also makeSliceView and other operations are needed especially - slice aware OPs
            // TODO: like in (i,j,k,l) tensor slicing, offset + new_stride[0]*i + new_stride[1]*(i+1) + ... + new_stride[n-1]*(i+n-1) will be required to view the slice data values and the same is needed for their ops
            // TODO: Interesting to figure out how ops would work
             return SliceReturn(shape, strides, offset); // TODO: remove this line - boilerplace
            // TODO: complete this as this can be used by the slice function inside the Tensor class to compute the slice

        }



    }
}
