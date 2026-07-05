#include "slice.h"

namespace bearml {
    namespace utils {


        SliceReturn computing_slice_parameters(std::vector<int> shape, std::vector<int> strides, std::vector<utils::Slice> slices) {
            // we will have to traverse in reverse order to compute the offsets and end per dimension
            int offset = 0;
            std::vector<int> new_shape(shape.size());
            std::vector<int> new_strides(strides.size());
            for (size_t i = 0; i < slices.size(); i++) {
                int start = slices[i].get_set_start();
                int end = slices[i].get_set_end();
                int step = slices[i].get_set_step();
                offset += strides[i] * start;
                new_shape[i] = static_cast<int>(ceil((end - start) / (double)step));
                new_strides[i] = strides[i] * step;
            }

            return SliceReturn(new_shape, new_strides, offset);
            // One thing I learned is that for n-dim slicing to function offset is needed but also makeSliceView and other operations are needed especially - slice aware OPs
            // like in (i,j,k,l) tensor slicing, offset + new_stride[0]*i + new_stride[1]*(i+1) + ... + new_stride[n-1]*(i+n-1) will be required to view the slice data values and the same is needed for their ops
            // Interesting to figure out how ops would work
            // complete this as this can be used by the slice function inside the Tensor class to compute the slice
        }
    }
}
