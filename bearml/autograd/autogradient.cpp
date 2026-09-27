#include "autogradient.h"
#include "tensor/Tensor.h"

using ll = long long;
namespace bearml {
    namespace autogradient{

        // A better way to get the n-th derivative is to:
        // grads[root] = seed
        // for node in topo_order:
        //     g = grads[node]                       // tracked if create_graph
        //     in_alias  = tracked aliases of node->inputs
        //     out_alias = tracked alias of node
        //     contribs  = grad_of(node->op, node->op_attr, g, in_alias, out_alias)
        //     for i: if input i needs grad: grads[input_i] = grads[input_i] + sum_to_size(contribs[i], shape)
        // return grads[wrt...]

       // no named namespace as we do not want topological sort to be called outside
       namespace{
          template <typename T>
          std::vector<std::shared_ptr<bearml::Node<T>>> topological_sort(std::shared_ptr<bearml::Node<T>> end_node){

             std::map<std::shared_ptr<bearml::Node<T>>, ll> child_counts;
             std::vector<std::shared_ptr<bearml::Node<T>>> stack;
             stack.push_back(end_node);

             while (!stack.empty()){
                std::shared_ptr<bearml::Node<T>> node = stack[stack.size()-1]; stack.pop_back();
                auto f = child_counts.find(node);
                if (f != child_counts.end()){
                   child_counts[node] += 1;
                }else{
                   child_counts[node] = 1;
                   stack.insert(stack.end(),node->inputs.begin(),node->inputs.end()); // node->inputs are the parents
                }
             }


             std::vector<std::shared_ptr<bearml::Node<T>>> childless_nodes {end_node}; // assumption is that we do not consider the child nodes of the calling node here, even if it has some children
             std::vector<std::shared_ptr<bearml::Node<T>>> sorted;
             while (!childless_nodes.empty()){
                std::shared_ptr<bearml::Node<T>> node = childless_nodes[childless_nodes.size()-1]; childless_nodes.pop_back();
                sorted.push_back(node); // like python yield
                for (auto parent  : node->inputs){

                   if (!parent) continue; // skip null

                   if (child_counts[parent] == 1){
                      childless_nodes.push_back(parent);
                   }else{
                      child_counts[parent] -= 1;
                   }
                }
             }

             return sorted;
          }
       }

       namespace{
          // true when every dimension is 1 (like {1} or {1,1}) - matmul treats these as a scalar
          template <typename T>
          bool is_scalar_shaped(const T& t) {
             const auto shape = t.getShape();
             return std::all_of(shape.begin(), shape.end(), [](int d) { return d == 1; });
          }
       }


        template <typename T>
        T compute_grad_for_mean(Node<T>& node, Node<T>& node_input) {
            ll n = node_input.val.sizeOfTensor();
            T grad_broadcast(node_input.val.getShape(), node_input.val.getDevice());
            // broadcasting spreads grad/n over every element - no copy to the cpu needed
            return grad_broadcast + node.grad * (1.0 / static_cast<double>(n));
        }


        // will be our function map - PROBLEM - does recomputation
        // ALSO problem - does not deal with double diff
        template <typename T>
        std::vector<T> grad_of(Node<T>& node) {
            switch (node.op) {
                case OP_Code::NO_OP:
                    return {}; // leaf node
                case OP_Code::OP_ADD:
                    return {node.grad, node.grad};
                case OP_Code::OP_SUB:
                    return {node.grad, node.grad*-1.0};
                case OP_Code::OP_MUL:
                    // scalar-shaped operand - the product is just a scaling
                    if (is_scalar_shaped(node.inputs[0]->val)) return {bearml::linear_algebra::hadamard(node.grad, node.inputs[1]->val), node.grad*node.inputs[0]->val};
                    if (is_scalar_shaped(node.inputs[1]->val)) return {node.grad*node.inputs[1]->val, bearml::linear_algebra::hadamard(node.grad, node.inputs[0]->val)};
                    // transpose needs a rank >= 2 tensor
                    if (node.inputs[0]->val.getShape().size() < 2 || node.inputs[1]->val.getShape().size() < 2)
                        throw std::logic_error("Autograd: matmul backward with a 1-D operand is not supported yet");
                    // grad_a = grad * b^T
                    // grad_b = a^T * grad
                    return {node.grad*node.inputs[1]->val.transpose(), node.inputs[0]->val.transpose()*node.grad};
                case OP_Code::OP_DIV:
                    // c = a/b
                    // dc/da = grad * (1/b)
                    // dc/db = grad * (-a/b^2)
                    return {bearml::linear_algebra::hadamard(node.grad, 1.0/node.inputs[1]->val), bearml::linear_algebra::hadamard(node.grad, -1.0 * node.inputs[0]->val / bearml::linear_algebra::hadamard(node.inputs[1]->val, node.inputs[1]->val))};
                case OP_Code::OP_MAX:
                    // grad_a = grad * (a >= b)
                    // grad_b = grad * (b >= a)
                    return {bearml::linear_algebra::hadamard(node.grad, bearml::linear_algebra::mask_of_greater_than_equal_to(node.inputs[0]->val,node.inputs[1]->val)), bearml::linear_algebra::hadamard(node.grad,  bearml::linear_algebra::mask_of_greater_than_equal_to(node.inputs[1]->val,node.inputs[0]->val))};
                case OP_Code::OP_MIN:
                    // grad_a = grad * (a <= b)
                    // grad_b = grad * (b <= a)
                    return {bearml::linear_algebra::hadamard(node.grad, bearml::linear_algebra::mask_of_less_than_equal_to(node.inputs[0]->val,node.inputs[1]->val)), bearml::linear_algebra::hadamard(node.grad,  bearml::linear_algebra::mask_of_less_than_equal_to(node.inputs[1]->val,node.inputs[0]->val))};
                case OP_Code::OP_HADAMARD:
                    // grad_a = grad * b
                    // grad_b = grad * a
                    return {bearml::linear_algebra::hadamard(node.grad, node.inputs[1]->val), bearml::linear_algebra::hadamard(node.grad, node.inputs[0]->val)};

                // UNARY OPS

                case OP_Code::OP_EXP:
                    return {bearml::linear_algebra::hadamard(node.grad, node.val)};
                case OP_Code::OP_SIN:
                    return {bearml::linear_algebra::hadamard(node.grad, T::cos(node.inputs[0]->val))};
                case OP_Code::OP_COS:
                    return {bearml::linear_algebra::hadamard(node.grad, -1.0 * T::sin(node.inputs[0]->val))};
                case OP_Code::OP_TAN:
                    // grad_tan = grad * (1 + tan^2) [Note -> sec^2 = 1 + tan^2]
                    return {bearml::linear_algebra::hadamard(node.grad, 1.0 +  bearml::linear_algebra::hadamard(T::tan(node.inputs[0]->val), T::tan(node.inputs[0]->val)))};
                case OP_Code::OP_SINH:
                    return {bearml::linear_algebra::hadamard(node.grad, T::cosh(node.inputs[0]->val))};
                case OP_Code::OP_COSH:
                    return {bearml::linear_algebra::hadamard(node.grad, T::sinh(node.inputs[0]->val))};
                case OP_Code::OP_TANH:
                    return {bearml::linear_algebra::hadamard(node.grad, 1.0 - bearml::linear_algebra::hadamard(node.val,node.val))};
                case OP_Code::OP_TRANSPOSE:
                    return {node.grad.transpose()};
                case OP_Code::OP_ABS:
                    return {bearml::linear_algebra::hadamard(node.grad, bearml::linear_algebra::sign(node.inputs[0]->val))};
                case OP_Code::OP_LOG:
                    return {bearml::linear_algebra::hadamard(node.grad, 1.0 / node.inputs[0]->val)};
                case OP_Code::OP_SQRT:
                    return {bearml::linear_algebra::hadamard(node.grad, 1.0 / (2.0 * T::sqrt(node.inputs[0]->val)))};
                case OP_Code::OP_MEAN_FOR_GRAD:
                    // c = mean(a)
                    // c = 1/n * sum(a)
                    // dL/da = dL/dc * dc/da (sum has gradient as 1 so only 1/n remains)
                    return {compute_grad_for_mean(node, *node.inputs[0])};
                case OP_Code::OP_PAD:
                    return {bearml::neural_network::padding(node.grad, -node.op_attr.pad_amount, node.op_attr.pad_mode)};
                // SCALAR OPS - one tensor input, constant in op_attr

                case OP_Code::OP_ADD_SCALAR:
                    // c is constant so it drops out
                    return {node.grad};
                case OP_Code::OP_SUB_SCALAR:
                    // a - c
                    return {node.grad};
                case OP_Code::OP_RSUB_SCALAR:
                    // c - a
                    return {-1.0 * node.grad};
                case OP_Code::OP_MUL_SCALAR:
                    return {node.op_attr.constant * node.grad};
                case OP_Code::OP_DIV_SCALAR:
                    // a / c
                    return {node.grad / node.op_attr.constant};
                case OP_Code::OP_RDIV_SCALAR:
                    // c / a  =>  dc/da = -c / a^2
                    return {bearml::linear_algebra::hadamard(node.grad, -node.op_attr.constant / bearml::linear_algebra::hadamard(node.inputs[0]->val, node.inputs[0]->val))};

                case OP_Code::OP_SOFTMAX: {
                    // grad_a = y * (grad - sum_dim(grad * y)) where y is the softmax output
                    T grad_times_output = bearml::linear_algebra::hadamard(node.grad, node.val);
                    T row_sum = grad_times_output.accumulate(node.op_attr.dim, bearml::reductions::ReductionOps::SUM, true);
                    return {bearml::linear_algebra::hadamard(node.val, node.grad - row_sum)};
                }
                case OP_Code::OP_SUM: {
                    // every summed element gets grad - broadcast back over the reduced dim
                    T grad_broadcast(node.inputs[0]->val.getShape(), node.inputs[0]->val.getDevice());
                    return {grad_broadcast + node.grad};
                }
                default:
                    throw std::invalid_argument("Autograd: OP Code does not exist or not implemented yet!");
            }
        };

        template<typename T>
        void accumulate_grad(T& target_grad, const T& grad_contribution, const T& target_val) {
            auto target_grad_shape = target_val.getShape();
            target_grad = target_grad +  bearml::linear_algebra::reduce(grad_contribution, target_grad_shape, reduction_op);
        }

        template<typename T>
        void apply_grad(Node<T>& node) {
            const OP_ARITY arity = op_arity(node.op);
            if (node.inputs.size() != static_cast<size_t>(arity))
                throw std::logic_error("Autograd: input count does not match op arity (Binary op requires 2 inputs, Unary op requires 1 input and so on)");

            auto grads = grad_of(node);// pushes n.grad into its inputs
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                if (!node.inputs[i]->requires_grad) continue; // constant inputs don't get a gradient
                accumulate_grad(node.inputs[i]->grad, grads[i], node.inputs[i]->val);
            }
        }

       // TODO: fix for MATRIX AND TENSOR TYPES - also add a boolean for gradient accumulation as well
       template <typename T>
       T backward(std::shared_ptr<bearml::Node<T>> end_node, bool accumulate){

          std::vector<std::shared_ptr<bearml::Node<T>>> all_nodes = topological_sort(end_node);
          // we clear grads right now
          if (!accumulate){
             for (const auto& node : all_nodes) {
                if (node->requires_grad){
                    node->grad = T(node->val.getShape(), node->val.getDevice());
                }
             }
          } else {
             // only leaves accumulate - intermediate grads still restart, or they would be counted twice
             for (const auto& node : all_nodes) {
                if (node->requires_grad && node->op != OP_Code::NO_OP){
                    node->grad = T(node->val.getShape(), node->val.getDevice());
                }
             }
          }

          end_node->grad = T(end_node->val.getShape(), end_node->val.getDevice()); // The whole matrix is filled with 1 - because we want to compute the Jacobian matrix
          end_node->grad.fill(1.0); // fill returns void;

          for (const auto& node : all_nodes) {
             // Only propagate from nodes made by an op that needs a gradient
             if (node->op != OP_Code::NO_OP && node->requires_grad) {
                   apply_grad(*node);
             }
          }
          return end_node->grad;
       }

       // backward from a tracked tensor rather than from its node
       template <typename E>
       Tensor<E> backward(const Tensor<E>& root, bool accumulate) {
          const auto& node = TensorAccess::node(root);
          if (!node) throw std::logic_error("backward: root tensor is not tracked");
          return backward(node, accumulate);
       }

       // runs backward from root and collects the gradient of each tensor in wrt
       template <typename E>
       std::vector<Tensor<E>> gradients(const Tensor<E>& root, std::initializer_list<const Tensor<E>*> wrt) {
          // tensors the root doesn't reach would otherwise keep an old gradient
          for (const Tensor<E>* t : wrt) {
             const auto& node = TensorAccess::node(*t);
             if (node) node->grad.fill(E(0));
          }
          backward(root);
          std::vector<Tensor<E>> grads;
          grads.reserve(wrt.size());
          for (const Tensor<E>* t : wrt) grads.push_back(t->grad());
          return grads;
       }
    }

    // ------------------------------ Tensor autograd members ------------------------------

    // makes a node for output when any input is tracked - untracked inputs become constant nodes
    template <typename E>
    void Tensor<E>::record_op(OP_Code op, std::initializer_list<const Tensor*> inputs, OpAttributes attrs, Tensor& output) {
       if constexpr (!is_floating(dtype_of<E>)) {
          (void)op; (void)inputs; (void)attrs; (void)output; // integer tensors are never tracked
       } else {
          bool tracked = false;
          for (const Tensor* in : inputs) {
             if (in->graph_node) { tracked = true; break; }
          }
          if (!tracked) return;

          auto node = std::make_shared<Node<Tensor>>();
          node->val = output.detach();
          node->grad = Tensor(output.getShape(), output.getDevice());
          node->op = op;
          node->op_attr = attrs;
          node->requires_grad = true;
          node->inputs.reserve(inputs.size());
          for (const Tensor* in : inputs) {
             if (in->graph_node) {
                // leaf (e.g. W): to_ or *= may have given it new memory, so point val at the current memory
                if (in->graph_node->op == OP_Code::NO_OP) in->graph_node->val = in->detach();
                node->inputs.push_back(in->graph_node);
             } else {
                auto constant = std::make_shared<Node<Tensor>>(); // keeps inputs lined up with op_arity
                constant->val = in->detach();
                node->inputs.push_back(constant);
             }
          }
          output.graph_node = node;
       }
    }

    // in-place ops don't record, so they are only allowed on leaves and with untracked operands
    template <typename E>
    void Tensor<E>::check_inplace(const Tensor* other) const {
       if (graph_node && graph_node->op != OP_Code::NO_OP) {
          throw std::logic_error("in-place op on the result of a tracked op; use the out-of-place op instead");
       }
       if (other && other->graph_node) {
          throw std::logic_error("in-place op with a tracked operand; use the out-of-place op instead");
       }
    }

    // turns this tensor into a leaf of the graph (or drops it from the graph)
    template <typename E>
    void Tensor<E>::set_requires_grad(bool on) {
       if constexpr (!is_floating(dtype_of<E>)) {
          if (on) throw std::logic_error("set_requires_grad: autograd needs a floating-point tensor");
       } else {
          if (!on) { graph_node = nullptr; return; }
          if (graph_node) return; // already tracked
          auto node = std::make_shared<Node<Tensor>>();
          node->val = detach();
          node->grad = Tensor(getShape(), getDevice());
          node->requires_grad = true;
          graph_node = node;
       }
    }

    template <typename E>
    bool Tensor<E>::requires_grad() const {
       return graph_node != nullptr;
    }

    template <typename E>
    const Tensor<E>& Tensor<E>::grad() const {
       if (!graph_node) throw std::logic_error("grad: tensor is not tracked");
       return graph_node->grad;
    }

    template <typename E>
    void Tensor<E>::zero_grad() {
       if (graph_node) graph_node->grad.fill(E(0));
    }
}

// Template specified

#define BEARML_AUTOGRAD_TENSOR_MEMBERS(E) \
    template void bearml::Tensor<E>::record_op(OP_Code, std::initializer_list<const bearml::Tensor<E>*>, OpAttributes, bearml::Tensor<E>&); \
    template void bearml::Tensor<E>::check_inplace(const bearml::Tensor<E>*) const; \
    template void bearml::Tensor<E>::set_requires_grad(bool); \
    template bool bearml::Tensor<E>::requires_grad() const; \
    template const bearml::Tensor<E>& bearml::Tensor<E>::grad() const; \
    template void bearml::Tensor<E>::zero_grad();

#define BEARML_AUTOGRAD_ENGINE(E) \
    template bearml::Tensor<E> bearml::autogradient::compute_grad_for_mean<bearml::Tensor<E>>(bearml::Node<bearml::Tensor<E>>&, bearml::Node<bearml::Tensor<E>>&); \
    template std::vector<bearml::Tensor<E>> bearml::autogradient::grad_of<bearml::Tensor<E>>(bearml::Node<bearml::Tensor<E>>&); \
    template void bearml::autogradient::accumulate_grad<bearml::Tensor<E>>(bearml::Tensor<E>&, const bearml::Tensor<E>&, const bearml::Tensor<E>&); \
    template void bearml::autogradient::apply_grad<bearml::Tensor<E>>(bearml::Node<bearml::Tensor<E>>&); \
    template bearml::Tensor<E> bearml::autogradient::backward<bearml::Tensor<E>>(std::shared_ptr<bearml::Node<bearml::Tensor<E>>>, bool); \
    template bearml::Tensor<E> bearml::autogradient::backward<E>(const bearml::Tensor<E>&, bool); \
    template std::vector<bearml::Tensor<E>> bearml::autogradient::gradients<E>(const bearml::Tensor<E>&, std::initializer_list<const bearml::Tensor<E>*>);

BEARML_AUTOGRAD_TENSOR_MEMBERS(float)
BEARML_AUTOGRAD_TENSOR_MEMBERS(double)
BEARML_AUTOGRAD_TENSOR_MEMBERS(int)
BEARML_AUTOGRAD_ENGINE(float)
BEARML_AUTOGRAD_ENGINE(double)
#if defined(__STDCPP_BFLOAT16_T__)
BEARML_AUTOGRAD_TENSOR_MEMBERS(std::bfloat16_t)
BEARML_AUTOGRAD_ENGINE(std::bfloat16_t)
#endif
