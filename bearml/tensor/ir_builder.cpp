#include "ir_builder.h"
#include <stdexcept>

namespace bearml {

    InferredType infer_type(OP_Code op, const std::vector<std::shared_ptr<IRNode>>& inputs, const OpAttributes& attrs) {
        throw std::logic_error("not implemented: infer_type");
    }

    bool attrs_equal(const OpAttributes& a, const OpAttributes& b) {
        throw std::logic_error("not implemented: attrs_equal");
    }

    std::shared_ptr<IRNode> IRBuilder::parameter(AnyTensor value) {
        throw std::logic_error("not implemented: IRBuilder::parameter");
    }

    std::shared_ptr<IRNode> IRBuilder::constant(AnyTensor value) {
        throw std::logic_error("not implemented: IRBuilder::constant");
    }

    std::shared_ptr<IRNode> IRBuilder::op(OP_Code op, std::vector<std::shared_ptr<IRNode>> inputs, OpAttributes attrs) {
        throw std::logic_error("not implemented: IRBuilder::op");
    }

    size_t IRBuilder::num_nodes() const {
        throw std::logic_error("not implemented: IRBuilder::num_nodes");
    }

    void IRBuilder::clear() {
        throw std::logic_error("not implemented: IRBuilder::clear");
    }

    bool IRBuilder::NodeKey::operator==(const NodeKey& other) const {
        throw std::logic_error("not implemented: IRBuilder::NodeKey::operator==");
    }

    size_t IRBuilder::NodeKeyHash::operator()(const NodeKey& k) const {
        throw std::logic_error("not implemented: IRBuilder::NodeKeyHash");
    }

    // one builder per thread - a function-local thread_local is created on first use
    IRBuilder& default_builder() {
        thread_local IRBuilder builder;
        return builder;
    }
}
