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
        // TODO: Design these structures
        Shape shape;
        DType dtype;
        // Attrs attrs;

    };
};
