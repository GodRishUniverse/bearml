#pragma once
#include <vector>
#include <memory>
#include "operators/ops.h"
#include "utils/shape_utils.h"
#include "utils/dtype.h"

namespace bearml {

    struct IRNode {
        OP_Code op;
        std::vector<std::shared_ptr<IRNode>> inputs;
        Shape shape;
        DType dtype;
        OpAttributes attrs;
    };
};
