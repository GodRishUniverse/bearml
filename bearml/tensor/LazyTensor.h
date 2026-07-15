// We will create a tensor class (will see if we inherit from Tensor class or not) to be compiled
// TODO: Help with graph keeping and operator fusing of ops
// TODO: Can use XLA for faster execution as everything is precompiled
// TODO: need to do more research into XLA/MLIR and ML compilers and how they can be used to compile tensor ops

#pragma once
#include "Tensor.h"
#include "operators/ops.h"
#include <memory>
#include <vector>

// TODO: need to think about how to implement this and design this so that we can use it correctly especially - it should mirror Tensor EXACTLY
namespace bearml {

    // Uh Idk I need to design this properly
    // utility enum to distinguish between tensor nodes and op nodes
    enum class LazyTensorNodeType {
        TENSOR,
        OP
    };

    // TODO: check how to implement this and design this (the graph)
    template<typename T>
    struct LazyTensorMetadata{
        std::vector<int> shape; // will be used to store the mem alloc needed
        std::vector<int> strides;
        Device device;
        size_t data_offset;
        bool is_sliced_view;
    };

    template <typename T>
    class LazyTensorNode {
        public:
            // we need incoming op and output op to know how to fuse
            LazyTensorNodeType node_type; // it can either be a tensor node or an op node
            std::string node_name; // the name of the node (for debugging purposes) - like can be variable name (if tensor) or op name ("add_1", "mul_2", etc.) - numbers are in topological order
            OP_Code op_code = OP_Code::NO_OP ; // all nodes are initially NO_OP - only set once node_type is known
            // TODO: I also need to extend my ops enum
            std::vector<std::shared_ptr<LazyTensorNode<T>>> inputs; // get the inputs to this op so that we can materialize it
            LazyTensorMetadata<T> metadata;

            //
            Tensor<T> materialize(); // materialize means to compute the tensor from the inputs
    };
}
