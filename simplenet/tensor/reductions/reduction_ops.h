#pragma once
#include <cstdint>


namespace simplenet {
    class Tensor; // forward declaration
    namespace reductions {
        enum class ReductionOps: uint16_t {
            SUM,
            MEAN,
            MAX,
            MIN,
            PROD,
            ARG_MAX,
            ARG_MIN,
            // TODO: add remaining reduction ops
        };

    }
}
