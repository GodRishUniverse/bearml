#pragma once

#ifndef STRIDED_ORDERING_H
#define STRIDED_ORDERING_H

namespace simplenet {
    namespace utils {


        // Memory layout descriptor for a Tensor's current strides relative to its shape.
        // - ROW_MAJOR : strides match row-major (C order) AND data_offset == 0 (cause no strides)
        // - COL_MAJOR : strides match col-major (Fortran order) AND data_offset == 0 (cause no strides)
        // - STRIDED  : neither — e.g. broadcast view, slice view, permuted view, offset != 0
        enum class Layout { ROW_MAJOR, COL_MAJOR, STRIDED };

    } // namespace utils
} // namespace simplenet

#endif // STRIDED_ORDERING_H
