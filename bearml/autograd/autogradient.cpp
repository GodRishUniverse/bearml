#include "autogradient.h"
#include "tensor/Tensor.h"

using ll = long long;
namespace bearml {
    namespace autogradient{

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

       namespace {
          // shapes where every dimension is 1: matmul treats these as scalars
          template <typename T>
          bool is_scalar_shaped(const T& t) {
             const auto shape = t.getShape();
             return std::all_of(shape.begin(), shape.end(), [](int d) { return d == 1; });
          }

          // sums a contribution back to the target's shape, then adds it
          template <typename T>
          void accumulate_grad(T& target_grad, const T& contribution, const T& target_val) {
             std::vector<int> target_shape = target_val.getShape();
             target_grad += bearml::linear_algebra::reduce(contribution, target_shape, reduction_op);
          }

          template <typename T>
          T run_backward(std::shared_ptr<bearml::Node<T>> end_node, const T* seed, bool accumulate) {
             std::vector<std::shared_ptr<bearml::Node<T>>> all_nodes = topological_sort(end_node);

             for (const auto& node : all_nodes) {
                if (!node->requires_grad) continue;
                if (!accumulate || node->grad.const_data() == nullptr) {
                   node->grad = T(node->val.getShape(), node->val.getDevice());
                }
             }

             if (seed) {
                if (seed->getShape() != end_node->val.getShape()) {
                   throw std::invalid_argument("backward: seed shape does not match the root");
                }
                end_node->grad = *seed;
             } else {
                end_node->grad = T(end_node->val.getShape(), end_node->val.getDevice());
                end_node->grad.fill(1.0);
             }

             for (const auto& node : all_nodes) {
                if (node->op != OP_Code::NO_OP && node->requires_grad) {
                   apply_grad(*node);
                }
             }
             return end_node->grad;
          }
       }

       template <typename T>
       std::vector<T> grad_of(Node<T>& node) {
          T& g = node.grad;
          switch (node.op) {
             case OP_Code::NO_OP:
                return {}; // leaf node

             // BINARY OPS

             case OP_Code::OP_ADD:
                return {g, g};
             case OP_Code::OP_SUB:
                return {g, g * -1.0};
             case OP_Code::OP_MUL: {
                T& a = node.inputs[0]->val;
                T& b = node.inputs[1]->val;
                // scalar-shaped operand: the product is elementwise scaling
                if (is_scalar_shaped(a)) return {bearml::linear_algebra::hadamard(g, b), g * a};
                if (is_scalar_shaped(b)) return {g * b, bearml::linear_algebra::hadamard(g, a)};
                if (a.getShape().size() < 2 || b.getShape().size() < 2) {
                   throw std::logic_error("Autograd: matmul backward with a 1-D operand is not supported yet");
                }
                // grad_a = grad * b^T, grad_b = a^T * grad
                return {g * b.transpose(), a.transpose() * g};
             }
             case OP_Code::OP_DIV: {
                // c = a/b: dc/da = 1/b, dc/db = -a/b^2
                T& a = node.inputs[0]->val;
                T& b = node.inputs[1]->val;
                return {bearml::linear_algebra::hadamard(g, 1.0 / b),
                        bearml::linear_algebra::hadamard(g, -1.0 * a / bearml::linear_algebra::hadamard(b, b))};
             }
             case OP_Code::OP_MAX: {
                // grad flows to whichever input is the max (both on a tie)
                T& a = node.inputs[0]->val;
                T& b = node.inputs[1]->val;
                return {bearml::linear_algebra::hadamard(g, bearml::linear_algebra::mask_of_greater_than_equal_to(a, b)),
                        bearml::linear_algebra::hadamard(g, bearml::linear_algebra::mask_of_greater_than_equal_to(b, a))};
             }
             case OP_Code::OP_MIN: {
                T& a = node.inputs[0]->val;
                T& b = node.inputs[1]->val;
                return {bearml::linear_algebra::hadamard(g, bearml::linear_algebra::mask_of_less_than_equal_to(a, b)),
                        bearml::linear_algebra::hadamard(g, bearml::linear_algebra::mask_of_less_than_equal_to(b, a))};
             }
             case OP_Code::OP_HADAMARD:
                return {bearml::linear_algebra::hadamard(g, node.inputs[1]->val),
                        bearml::linear_algebra::hadamard(g, node.inputs[0]->val)};

             // UNARY OPS - node.val is the output, inputs[0]->val the input

             case OP_Code::OP_EXP:
                return {bearml::linear_algebra::hadamard(g, node.val)};
             case OP_Code::OP_SIN:
                return {bearml::linear_algebra::hadamard(g, T::cos(node.inputs[0]->val))};
             case OP_Code::OP_COS:
                return {bearml::linear_algebra::hadamard(g, -1.0 * T::sin(node.inputs[0]->val))};
             case OP_Code::OP_TAN: {
                // sec^2 = 1 + tan^2
                T t = T::tan(node.inputs[0]->val);
                return {bearml::linear_algebra::hadamard(g, 1.0 + bearml::linear_algebra::hadamard(t, t))};
             }
             case OP_Code::OP_SINH:
                return {bearml::linear_algebra::hadamard(g, T::cosh(node.inputs[0]->val))};
             case OP_Code::OP_COSH:
                return {bearml::linear_algebra::hadamard(g, T::sinh(node.inputs[0]->val))};
             case OP_Code::OP_TANH:
                return {bearml::linear_algebra::hadamard(g, 1.0 - bearml::linear_algebra::hadamard(node.val, node.val))};
             case OP_Code::OP_TRANSPOSE:
                return {g.transpose()};
             case OP_Code::OP_ABS:
                return {bearml::linear_algebra::hadamard(g, bearml::linear_algebra::sign(node.inputs[0]->val))};
             case OP_Code::OP_LOG:
                return {bearml::linear_algebra::hadamard(g, 1.0 / node.inputs[0]->val)};
             case OP_Code::OP_SQRT:
                return {bearml::linear_algebra::hadamard(g, 1.0 / (2.0 * T::sqrt(node.inputs[0]->val)))};
             case OP_Code::OP_MEAN_FOR_GRAD: {
                // c = 1/n * sum(a): every element gets g/n
                T& a = node.inputs[0]->val;
                T zeros(a.getShape(), a.getDevice());
                return {zeros + g * (1.0 / static_cast<double>(a.sizeOfTensor()))};
             }
             case OP_Code::OP_SUM: {
                // every summed element gets g, broadcast back over the reduced axis
                T& a = node.inputs[0]->val;
                T zeros(a.getShape(), a.getDevice());
                return {zeros + g};
             }
             case OP_Code::OP_SOFTMAX: {
                // grad_a = y * (g - sum_dim(g * y)), y = softmax output
                T gy = bearml::linear_algebra::hadamard(g, node.val);
                T row_sum = gy.accumulate(node.op_attr.dim, bearml::reductions::ReductionOps::SUM, true);
                return {bearml::linear_algebra::hadamard(node.val, g - row_sum)};
             }
             case OP_Code::OP_PAD:
                return {bearml::neural_network::padding(g, -node.op_attr.pad_amount, node.op_attr.pad_mode)};

             // SCALAR OPS - one tensor input, constant in op_attr

             case OP_Code::OP_ADD_SCALAR:
                return {g};
             case OP_Code::OP_SUB_SCALAR:
                // a - c
                return {g};
             case OP_Code::OP_RSUB_SCALAR:
                // c - a
                return {-1.0 * g};
             case OP_Code::OP_MUL_SCALAR:
                return {node.op_attr.constant * g};
             case OP_Code::OP_DIV_SCALAR:
                // a / c
                return {g / node.op_attr.constant};
             case OP_Code::OP_RDIV_SCALAR: {
                // c / a  =>  -c / a^2
                T& a = node.inputs[0]->val;
                return {bearml::linear_algebra::hadamard(g, -node.op_attr.constant / bearml::linear_algebra::hadamard(a, a))};
             }
             default:
                throw std::invalid_argument("Autograd: OP Code does not exist or not implemented yet!");
          }
       }

       template <typename T>
       void apply_grad(Node<T>& node) {
          if (node.inputs.size() != static_cast<size_t>(op_arity(node.op))) {
             throw std::logic_error("Autograd: input count does not match op arity");
          }
          std::vector<T> grads = grad_of(node);
          for (size_t i = 0; i < node.inputs.size(); ++i) {
             Node<T>& input = *node.inputs[i];
             if (!input.requires_grad) continue; // constants get no gradient
             accumulate_grad(input.grad, grads[i], input.val);
          }
       }

       template <typename T>
       T backward(std::shared_ptr<bearml::Node<T>> end_node, bool accumulate){
          return run_backward<T>(end_node, nullptr, accumulate);
       }

       template <typename E>
       Tensor<E> backward(const Tensor<E>& root, bool accumulate) {
          const auto& node = TensorAccess::node(root);
          if (!node) throw std::logic_error("backward: root tensor is not tracked");
          return run_backward<Tensor<E>>(node, nullptr, accumulate);
       }

       template <typename E>
       Tensor<E> backward(const Tensor<E>& root, const Tensor<E>& seed, bool accumulate) {
          const auto& node = TensorAccess::node(root);
          if (!node) throw std::logic_error("backward: root tensor is not tracked");
          return run_backward<Tensor<E>>(node, &seed, accumulate);
       }

       template <typename E>
       std::vector<Tensor<E>> gradients(const Tensor<E>& root, std::initializer_list<const Tensor<E>*> wrt) {
          backward(root);
          std::vector<Tensor<E>> grads;
          grads.reserve(wrt.size());
          for (const Tensor<E>* t : wrt) grads.push_back(t->grad());
          return grads;
       }
    }

    // ------------------------------ Tensor autograd members ------------------------------

    template <typename E>
    void Tensor<E>::record_op(OP_Code op, std::initializer_list<const Tensor*> inputs, OpAttributes attrs) {
       if constexpr (!is_floating(dtype_of<E>)) {
          (void)op; (void)inputs; (void)attrs; // integer tensors are never tracked
       } else {
          bool tracked = false;
          for (const Tensor* in : inputs) {
             if (in->graph_node) { tracked = true; break; }
          }
          if (!tracked) return;

          auto node = std::make_shared<Node<Tensor>>();
          node->val = detach();
          node->op = op;
          node->op_attr = attrs;
          node->requires_grad = true;
          node->inputs.reserve(inputs.size());
          for (const Tensor* in : inputs) {
             if (in->graph_node) {
                // a leaf's storage may have been replaced (to_, *=) since it was last used
                if (in->graph_node->op == OP_Code::NO_OP) in->graph_node->val = in->detach();
                node->inputs.push_back(in->graph_node);
             } else {
                auto constant = std::make_shared<Node<Tensor>>(); // keeps inputs aligned with op_arity
                constant->val = in->detach();
                node->inputs.push_back(constant);
             }
          }
          graph_node = node;
       }
    }

    template <typename E>
    void Tensor<E>::check_inplace(const Tensor* other) const {
       if (graph_node && graph_node->op != OP_Code::NO_OP) {
          throw std::logic_error("in-place op on the result of a tracked op; use the out-of-place op instead");
       }
       if (other && other->graph_node) {
          throw std::logic_error("in-place op with a tracked operand; use the out-of-place op instead");
       }
    }

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
       if (graph_node->grad.const_data() == nullptr) throw std::logic_error("grad: backward has not run for this tensor");
       return graph_node->grad;
    }

    template <typename E>
    void Tensor<E>::zero_grad() {
       if (graph_node && graph_node->grad.const_data() != nullptr) graph_node->grad.fill(E(0));
    }
}

// Template specified

#define BEARML_AUTOGRAD_TENSOR_MEMBERS(E) \
    template void bearml::Tensor<E>::record_op(OP_Code, std::initializer_list<const bearml::Tensor<E>*>, OpAttributes); \
    template void bearml::Tensor<E>::check_inplace(const bearml::Tensor<E>*) const; \
    template void bearml::Tensor<E>::set_requires_grad(bool); \
    template bool bearml::Tensor<E>::requires_grad() const; \
    template const bearml::Tensor<E>& bearml::Tensor<E>::grad() const; \
    template void bearml::Tensor<E>::zero_grad();

#define BEARML_AUTOGRAD_ENGINE(E) \
    template std::vector<bearml::Tensor<E>> bearml::autogradient::grad_of<bearml::Tensor<E>>(bearml::Node<bearml::Tensor<E>>&); \
    template void bearml::autogradient::apply_grad<bearml::Tensor<E>>(bearml::Node<bearml::Tensor<E>>&); \
    template bearml::Tensor<E> bearml::autogradient::backward<bearml::Tensor<E>>(std::shared_ptr<bearml::Node<bearml::Tensor<E>>>, bool); \
    template bearml::Tensor<E> bearml::autogradient::backward<E>(const bearml::Tensor<E>&, bool); \
    template bearml::Tensor<E> bearml::autogradient::backward<E>(const bearml::Tensor<E>&, const bearml::Tensor<E>&, bool); \
    template std::vector<bearml::Tensor<E>> bearml::autogradient::gradients<E>(const bearml::Tensor<E>&, std::initializer_list<const bearml::Tensor<E>*>);

BEARML_AUTOGRAD_TENSOR_MEMBERS(float)
BEARML_AUTOGRAD_TENSOR_MEMBERS(double)
BEARML_AUTOGRAD_TENSOR_MEMBERS(int)
BEARML_AUTOGRAD_ENGINE(float)
BEARML_AUTOGRAD_ENGINE(double)
