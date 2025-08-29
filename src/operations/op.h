#pragma once

#ifndef OPERATIONS
#define OPERATIONS

namespace simplenet {
    namespace Operations {
        struct ReductionOp {
               enum Type { FLATTEN, SUM } type;
               int startDim, endDim;
               bool keepdims;
        };
    }
}
#endif
