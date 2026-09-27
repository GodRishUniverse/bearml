#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include "operators/ops.h"
#include "utils/shape_utils.h"
#include "utils/dtype.h"

namespace bearml {

    struct IRNode {
        OP_Code op;
        std::vector<IRNode*> inputs;
        Shape shape;
        DType dtype;
        OpAttributes attrs;
    };
};
