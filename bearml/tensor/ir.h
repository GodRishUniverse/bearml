#pragma once
#include <vector>
#include <memory>
#include <string>
#include <variant>
#include "operators/ops.h"
#include "utils/shape_utils.h"
#include "utils/dtype.h"
#include "Tensor.h"


// We will use MLIR and stablehlo as the backend
namespace bearml {

    // std::variant is a type-safe union: holds exactly one of the listed types at a time (very cool feature as this knows the type unlike union only -> https://stackoverflow.com/questions/42082328/where-to-use-stdvariant-over-union)
    // std::visit runs code on whichever one it holds
    // leaf payload - a concrete tensor whose element type is only known at runtime
    using AnyTensor = std::variant<
        Tensor<float>,
        Tensor<double>,
        Tensor<int>
        #if defined(__STDCPP_BFLOAT16_T__)
            , Tensor<std::bfloat16_t> // only if cpu c++23 is being used
        #endif
    >;

    // IRNodeType: what a node in the graph is
    enum class IRNodeType: uint8_t {
        PARAMETER, // op parameter - input fed at run time (weights, a batch)
        CONSTANT,  // value baked into the graph
        OP         // computed from inputs
    };

    // one value in the graph - it describes a tensor that does not exist yet, so it has dims but no strides
    struct IRNode {
        IRNodeType node_type = IRNodeType::OP;        // leaf or op
        OP_Code op = OP_Code::NO_OP;                  // NO_OP for leaves
        std::vector<std::shared_ptr<IRNode>> inputs;  // the edges of the graph - empty for leaves
        std::vector<int> shape;                       // logical dims only - layout (strides) is decided when materializing
        DType dtype;                                  // element type, known at build time
        Device device;                                // where it will be computed
        OpAttributes op_attrs;                        // op parameters (dim, scalar constant, ...)
        std::shared_ptr<AnyTensor> leaf_value;        // real data - PARAMETER/CONSTANT only, keeps its own strides
        int id = -1;                                  // creation order - printed as %id in to_string
    };

    // This one line of IR text will be verbose like MLIR output e.g. "%3 = OP_ADD(%1, %2) : float32[4,8]" - leaves print as "%0 = parameter : float32[4,1]"
    std::string to_string(const IRNode& n);
};
