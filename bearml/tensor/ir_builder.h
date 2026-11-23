#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include "ir.h"

// TODO: this is boilerplate code at the moment
namespace bearml {

    // output type of an op - computed before the node exists so bad ops fail fast
    struct InferredType {
        std::vector<int> shape; // logical output dims
        DType dtype;            // output element type
        Device device;          // output device
    };

    // shape/dtype/device of an op's output - throws std::invalid_argument on shape, dtype or device mismatch
    // binary ops broadcast (reuse utils::computeBroadcastShape), *_SCALAR / unary / OP_SOFTMAX keep the shape,
    // OP_TRANSPOSE swaps the last two dims (Tensor::transpose), OP_SUM reduces attrs.dim with keepdims (Tensor::accumulate)
    InferredType infer_type(OP_Code op, const std::vector<std::shared_ptr<IRNode>>& inputs, const OpAttributes& attrs);

    // field-by-field equality - ops.h has no operator== for OpAttributes
    bool attrs_equal(const OpAttributes& a, const OpAttributes& b);

    // CSE = common subexpression elimination: an identical op over the same inputs is built once and reused
    // the only way to create IR nodes - assigns ids, infers types and merges duplicate ops (CSE)
    class IRBuilder {
        public:
            // new PARAMETER leaf holding value - never merged, two parameters are always two inputs
            std::shared_ptr<IRNode> parameter(AnyTensor value);

            // new CONSTANT leaf holding value - never merged
            std::shared_ptr<IRNode> constant(AnyTensor value);

            // op node over inputs - returns the existing node if an identical one is still alive, throws if infer_type does
            std::shared_ptr<IRNode> op(OP_Code op, std::vector<std::shared_ptr<IRNode>> inputs, OpAttributes attrs = {});

            // number of nodes created so far (leaves + ops, merged ops counted once)
            size_t num_nodes() const;

            // forget every node seen so far (after a sync point) - existing nodes stay valid
            void clear();

        private:
            // what makes two op nodes identical - inputs compared by identity (pointer), never by data
            struct NodeKey {
                OP_Code op;                        // op code
                std::vector<const IRNode*> inputs; // input node identities
                OpAttributes attrs;                // op parameters
                bool operator==(const NodeKey& other) const;
            };

            // combine op, input pointers and attrs into one hash
            struct NodeKeyHash {
                size_t operator()(const NodeKey& k) const;
            };

            // std::weak_ptr = non-owning pointer to a shared_ptr'd object; .lock() returns nullptr once the object is gone
            std::unordered_map<NodeKey, std::weak_ptr<IRNode>, NodeKeyHash> cse_table; // expired entry = miss
            int next_id = 0; // id given to the next node
    };

    // thread_local = one instance per thread, so two threads never share a CSE table
    // builder used by LazyTensor ops
    IRBuilder& default_builder();
}
