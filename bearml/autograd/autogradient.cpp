#include "autogradient.h"
#include "operators/ops.h"
#include "tensor/Tensor.h"
#include <vector>
#include <unordered_map>

using ll = long long;

// added for code clarity
template <typename e>
using beartens = bearml::Tensor<e>;

template <typename t>
using bearnode = bearml::Node<t>;

namespace bearml {
    namespace autogradient{

       // no named namespace as we do not want topological sort to be called outside
       namespace{
          template <typename T>
          std::vector<std::shared_ptr<bearnode<T>>> topological_sort(std::shared_ptr<bearnode<T>> end_node){

             std::map<std::shared_ptr<bearnode<T>>, ll> child_counts;
             std::vector<std::shared_ptr<bearnode<T>>> stack;
             stack.push_back(end_node);

             while (!stack.empty()){
                std::shared_ptr<bearnode<T>> node = stack[stack.size()-1]; stack.pop_back();
                auto f = child_counts.find(node);
                if (f != child_counts.end()){
                   child_counts[node] += 1;
                }else{
                   child_counts[node] = 1;
                   stack.insert(stack.end(),node->inputs.begin(),node->inputs.end()); // node->inputs are the parents
                }
             }


             std::vector<std::shared_ptr<bearnode<T>>> childless_nodes {end_node}; // assumption is that we do not consider the child nodes of the calling node here, even if it has some children
             std::vector<std::shared_ptr<bearnode<T>>> sorted;
             while (!childless_nodes.empty()){
                std::shared_ptr<bearnode<T>> node = childless_nodes[childless_nodes.size()-1]; childless_nodes.pop_back();
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
        T compute_grad_for_mean(const T& grad, const T& input) {
            ll n = input.sizeOfTensor();
            T grad_broadcast(input.getShape(), input.getDevice());
            // broadcasting spreads grad/n over every element - no copy to the cpu needed
            return grad_broadcast + grad * (1.0 / static_cast<double>(n));
        }

        // acts as a map of gradients (derivative lookup)
        // TODO: PROBLEM: WE HAVE RECOMPUTATION HERE - remove that
        template <typename T>
        std::vector<T> grad_of(OP_Code op, const OpAttributes& attr, const T& grad, const std::vector<T>& inputs, const T& output) {
            switch (op) {
                case OP_Code::NO_OP:
                    return {}; // leaf node
                case OP_Code::OP_ADD:
                    return {grad, grad};
                case OP_Code::OP_SUB:
                    return {grad, grad*-1.0};
                case OP_Code::OP_MUL:
                    // scalar-shaped operand - the product is just a scaling
                    // TODO: FIX - rank one matrix operand
                    if (is_scalar_shaped(inputs[0])) return {bearml::linear_algebra::hadamard(grad, inputs[1]), grad*inputs[0]};
                    if (is_scalar_shaped(inputs[1])) return {grad*inputs[1], bearml::linear_algebra::hadamard(grad, inputs[0])};
                    // transpose needs a rank >= 2 tensor
                    if (inputs[0].getShape().size() < 2 || inputs[1].getShape().size() < 2)
                        throw std::logic_error("Autograd: matmul backward with a 1-D operand is not supported yet");
                    // grad_a = grad * b^T
                    // grad_b = a^T * grad
                    return {grad*inputs[1].transpose(), inputs[0].transpose()*grad};
                case OP_Code::OP_DIV:
                    // c = a/b
                    // dc/da = grad * (1/b)
                    // dc/db = grad * (-a/b^2)
                    return {bearml::linear_algebra::hadamard(grad, 1.0/inputs[1]), bearml::linear_algebra::hadamard(grad, -1.0 * inputs[0] / bearml::linear_algebra::hadamard(inputs[1], inputs[1]))};
                case OP_Code::OP_MAX:
                    // grad_a = grad * (a >= b)
                    // grad_b = grad * (b >= a)
                    return {bearml::linear_algebra::hadamard(grad, bearml::linear_algebra::mask_of_greater_than_equal_to(inputs[0],inputs[1])), bearml::linear_algebra::hadamard(grad,  bearml::linear_algebra::mask_of_greater_than_equal_to(inputs[1],inputs[0]))};
                case OP_Code::OP_MIN:
                    // grad_a = grad * (a <= b)
                    // grad_b = grad * (b <= a)
                    return {bearml::linear_algebra::hadamard(grad, bearml::linear_algebra::mask_of_less_than_equal_to(inputs[0],inputs[1])), bearml::linear_algebra::hadamard(grad,  bearml::linear_algebra::mask_of_less_than_equal_to(inputs[1],inputs[0]))};
                case OP_Code::OP_HADAMARD:
                    // grad_a = grad * b
                    // grad_b = grad * a
                    return {bearml::linear_algebra::hadamard(grad, inputs[1]), bearml::linear_algebra::hadamard(grad, inputs[0])};

                // UNARY OPS

                case OP_Code::OP_EXP:
                    return {bearml::linear_algebra::hadamard(grad, output)};
                case OP_Code::OP_SIN:
                    return {bearml::linear_algebra::hadamard(grad, T::cos(inputs[0]))};
                case OP_Code::OP_COS:
                    return {bearml::linear_algebra::hadamard(grad, -1.0 * T::sin(inputs[0]))};
                case OP_Code::OP_TAN:
                    // grad_tan = grad * (1 + tan^2) [Note -> sec^2 = 1 + tan^2]
                    return {bearml::linear_algebra::hadamard(grad, 1.0 +  bearml::linear_algebra::hadamard(T::tan(inputs[0]), T::tan(inputs[0])))};
                case OP_Code::OP_SINH:
                    return {bearml::linear_algebra::hadamard(grad, T::cosh(inputs[0]))};
                case OP_Code::OP_COSH:
                    return {bearml::linear_algebra::hadamard(grad, T::sinh(inputs[0]))};
                case OP_Code::OP_TANH:
                    return {bearml::linear_algebra::hadamard(grad, 1.0 - bearml::linear_algebra::hadamard(output,output))};
                case OP_Code::OP_TRANSPOSE:
                    return {grad.transpose()};
                case OP_Code::OP_ABS:
                    return {bearml::linear_algebra::hadamard(grad, bearml::linear_algebra::sign(inputs[0]))};
                case OP_Code::OP_LOG:
                    return {bearml::linear_algebra::hadamard(grad, 1.0 / inputs[0])};
                case OP_Code::OP_SQRT:
                    return {bearml::linear_algebra::hadamard(grad, 1.0 / (2.0 * T::sqrt(inputs[0])))};
                case OP_Code::OP_MEAN_FOR_GRAD:
                    // c = mean(a)
                    // c = 1/n * sum(a)
                    // dL/da = dL/dc * dc/da (sum has gradient as 1 so only 1/n remains)
                    return {compute_grad_for_mean(grad, inputs[0])};
                case OP_Code::OP_PAD:
                    return {bearml::neural_network::padding(grad, -attr.pad_amount, attr.pad_mode)};
                // SCALAR OPS - one tensor input, constant in op_attr

                case OP_Code::OP_ADD_SCALAR:
                    // c is constant so it drops out
                    return {grad};
                case OP_Code::OP_SUB_SCALAR:
                    // a - c
                    return {grad};
                case OP_Code::OP_RSUB_SCALAR:
                    // c - a
                    return {-1.0 * grad};
                case OP_Code::OP_MUL_SCALAR:
                    return {attr.constant * grad};
                case OP_Code::OP_DIV_SCALAR:
                    // a / c
                    return {grad / attr.constant};
                case OP_Code::OP_RDIV_SCALAR:
                    // c / a  =>  dc/da = -c / a^2
                    return {bearml::linear_algebra::hadamard(grad, -attr.constant / bearml::linear_algebra::hadamard(inputs[0], inputs[0]))};

                case OP_Code::OP_SOFTMAX: {
                    // grad_a = y * (grad - sum_dim(grad * y)) where y is the softmax output
                    T grad_times_output = bearml::linear_algebra::hadamard(grad, output);
                    T row_sum = grad_times_output.accumulate(attr.dim, bearml::reductions::ReductionOps::SUM, true);
                    return {bearml::linear_algebra::hadamard(output, grad - row_sum)};
                }
                case OP_Code::OP_SUM: {
                    // every summed element gets grad - broadcast back over the reduced dim
                    T grad_broadcast(inputs[0].getShape(), inputs[0].getDevice());
                    return {grad_broadcast + grad};
                }
                default:
                    throw std::invalid_argument("Autograd: OP Code does not exist or not implemented yet!");
            }
        };

        template<typename T>
        void accumulate_grad(T& target_grad, const T& grad_contribution, const T& target_val) {
            std::vector<int> target_grad_shape = target_val.getShape();
            // add the reduced gradient contribution
            target_grad = target_grad +  bearml::linear_algebra::reduce(grad_contribution, target_grad_shape, reduction_op);
        }

        // We propogate the gradient here
        template<typename T>
        void apply_grad(
            const std::shared_ptr<Node<T>>& node,
            const T& upstream_grad,
            std::unordered_map<Node<T>*, T>& grad_map,
            bool create_graph
        ) {
            const OP_ARITY arity = op_arity(node->op);

            if (node->inputs.size() != static_cast<size_t>(arity)) {
                throw std::logic_error(
                    "Autograd: input count does not match op arity"
                );
            }

            std::vector<T> input_vals;
            input_vals.reserve(node->inputs.size());

            for (const auto& in : node->inputs) {
                if (create_graph && in->requires_grad) {
                    input_vals.push_back(in->val.autograd_tracked_view(in));
                } else {
                    input_vals.push_back(in->val);
                }
            }

            T output_val = node->val;

            if (create_graph) {
                output_val = node->val.autograd_tracked_view(node);
            }

            auto grads = grad_of(
                node->op,
                node->op_attr,
                upstream_grad,
                input_vals,
                output_val
            );

            for (size_t i = 0; i < node->inputs.size(); ++i) {
                auto& input = node->inputs[i];

                if (!input->requires_grad) {
                    continue;
                }

                auto it = grad_map.find(input.get());

                if (it == grad_map.end()) {
                    throw std::logic_error(
                        "Autograd: missing gradient-map entry"
                    );
                }

                accumulate_grad(
                    it->second,
                    grads[i],
                    input->val
                );
            }
        }

       // TODO: fix for MATRIX AND TENSOR TYPES - also add a boolean for gradient accumulation as well
       // backward_graph is needed in order to make the backward function differentiable as well in order to do higher order derivatives
       template <typename T>
       T backward(std::shared_ptr<bearnode<T>> end_node, bool accumulate,  bool create_graph){
          std::unordered_map<Node<T>*, T> grad_map;

          std::vector<std::shared_ptr<bearnode<T>>> all_nodes = topological_sort(end_node); // sort the graph
          // we clear grads right now
          for (const auto& node : all_nodes) {
              if (node->requires_grad) {
                  // insert_or_assign as operator[] needs Tensor's private default constructor
                  grad_map.insert_or_assign(node.get(), T(node->val.getShape(), node->val.getDevice()));
              }
          }

          grad_map.insert_or_assign(end_node.get(), T(
              end_node->val.getShape(),
              end_node->val.getDevice()
          )); // The whole matrix is filled with 1 - because we want to compute the Jacobian matrix

          grad_map.at(end_node.get()).fill(1.0);  // fill returns void;


          // walk in topological ordering
          for (const auto& node : all_nodes) {
             // Only propagate from nodes made by an op that needs a gradient
             if (node->op != OP_Code::NO_OP && node->requires_grad) {
                   apply_grad(node, grad_map.at(node.get()), grad_map, create_graph); // apply the gradient on that node
             }
          }

          // sync grad map to our node's definition
          for (const auto& node : all_nodes) {
              if (!node->requires_grad) {
                  continue;
              }

              auto it = grad_map.find(node.get());
              if (it == grad_map.end()) {
                  continue;
              }

              if (accumulate && node->op == OP_Code::NO_OP) {
                  node->grad = node->grad + it->second.detach();
              } else {
                  node->grad = it->second.detach();
              }
          }
          return end_node->grad;
       }

       // backward from a tracked tensor rather than from its node
       template <typename E>
       Tensor<E> backward(const Tensor<E>& root, bool accumulate, bool create_graph) {
          std::shared_ptr<Node<Tensor<E>>> node = root.graph_node;
          if (!node) throw std::logic_error("backward: root tensor is not tracked");
          return backward(node, accumulate, create_graph);
       }

       // Example -   auto g = gradients(loss, {&w, &b});   // g[0] = dloss/dw, g[1] = dloss/db
       // This is a different API than backward to get only particular gradients but still uses the backward function
       template <typename E>
       std::vector<Tensor<E>> gradients(const Tensor<E>& end_node, std::initializer_list<const Tensor<E>*> nodes_we_want_grads_with_respect_to) {
           // backward only resets nodes it reaches - zero these so an unreached tensor reads 0, not an old gradient
          for (const Tensor<E>* t : nodes_we_want_grads_with_respect_to) {
             const auto& node = t->graph_node;
             if (node) node->grad.fill(E(0));
          }
          // fills node->grad for everything end_node depends on
          backward(end_node); // compute the jacobian matrix

          std::vector<Tensor<E>> grads;
          grads.reserve(nodes_we_want_grads_with_respect_to.size()); // we want the gradients of no. of nodes in the nodes_we_want_grads_with_respect_to list

          for (const Tensor<E>* t : nodes_we_want_grads_with_respect_to){
              grads.push_back(t->grad());
          }
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
          // if neither input requires grads then we dont need to build autograd node so we avoid graph overhead here
          bool tracked = false;
          for (const Tensor* in : inputs) {
             if (in->graph_node) { tracked = true; break; }
          }
          if (!tracked) return;

          // Creating a node here
          auto node = std::make_shared<Node<Tensor>>();
          node->val = output.detach();
          node->grad = Tensor(output.getShape(), output.getDevice());
          node->op = op;
          node->op_attr = attrs;
          node->requires_grad = true;
          node->inputs.reserve(inputs.size());


          // we connect the node inputs now
          for (const Tensor* in : inputs) {
             if (in->graph_node) {
                if (in->graph_node->op == OP_Code::NO_OP) in->graph_node->val = in->detach(); // when we have no-op - we have a leaf node
                node->inputs.push_back(in->graph_node);
             } else {
                auto constant = std::make_shared<Node<Tensor>>(); // keeps inputs lined up with op_arity
                constant->val = in->detach();
                node->inputs.push_back(constant);
             }
          }
          output.graph_node = node; // attach created node to the output tensor
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

    // get the gradient of the current tensor
    template <typename E>
    const Tensor<E>& Tensor<E>::grad() const {
       if (!graph_node) throw std::logic_error("Autogradient: Tensor gradient is not tracked");
       return graph_node->grad;
    }

    template <typename E>
    void Tensor<E>::zero_grad() {
       if (graph_node) graph_node->grad.fill(E(0));
    }



    // A better way to get the n-th derivative is to:
    // grads[root] = seed
    // for node in topo_order:
    //     g = grads[node]                       // tracked if create_graph
    //     in_alias  = tracked aliases of node->inputs
    //     out_alias = tracked alias of node
    //     contribs  = grad_of(node->op, node->op_attr, g, in_alias, out_alias)
    //     for i:
    //         if input i needs grad:
    //             grads[input_i] = grads[input_i] + sum_to_size(contribs[i], shape)
    // return grads[wrt...]
    //
    //
    // Pytorch methodology -
    // def deriv(self, t, n=1):
    //      dx = self.forward(t)
    //      for i in range(n):
    //          dx = torch.autograd.grad(outputs=dx, inputs=t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    //      return dx
}

// Template specified

#define BEARML_AUTOGRAD_TENSOR_MEMBERS(E) \
    template void beartens<E>::record_op(OP_Code, std::initializer_list<const beartens<E>*>, OpAttributes, beartens<E>&); \
    template void beartens<E>::check_inplace(const beartens<E>*) const; \
    template void beartens<E>::set_requires_grad(bool); \
    template bool beartens<E>::requires_grad() const; \
    template const beartens<E>& beartens<E>::grad() const; \
    template void beartens<E>::zero_grad();

#define BEARML_AUTOGRAD_ENGINE(E) \
    template beartens<E> bearml::autogradient::compute_grad_for_mean<beartens<E>>(const beartens<E>&, const beartens<E>&); \
    template std::vector<beartens<E>> bearml::autogradient::grad_of<beartens<E>>(OP_Code op, const OpAttributes& attr, const beartens<E>& grad, const std::vector<beartens<E>>& inputs, const beartens<E>& output); \
    template void bearml::autogradient::accumulate_grad<beartens<E>>(beartens<E>&, const beartens<E>&, const beartens<E>&); \
    template void bearml::autogradient::apply_grad<beartens<E>>(const std::shared_ptr<bearml::Node<beartens<E>>>&, const beartens<E>&, std::unordered_map<bearml::Node<beartens<E>>*, beartens<E>>&, bool); \
    template beartens<E> bearml::autogradient::backward<beartens<E>>(std::shared_ptr<bearml::Node<beartens<E>>>, bool, bool); \
    template beartens<E> bearml::autogradient::backward<E>(const beartens<E>&, bool, bool); \
    template std::vector<beartens<E>> bearml::autogradient::gradients<E>(const beartens<E>&, std::initializer_list<const beartens<E>*>);

BEARML_AUTOGRAD_TENSOR_MEMBERS(float)
BEARML_AUTOGRAD_TENSOR_MEMBERS(double)
BEARML_AUTOGRAD_TENSOR_MEMBERS(int)
BEARML_AUTOGRAD_ENGINE(float)
BEARML_AUTOGRAD_ENGINE(double)
#if defined(__STDCPP_BFLOAT16_T__)
BEARML_AUTOGRAD_TENSOR_MEMBERS(std::bfloat16_t)
BEARML_AUTOGRAD_ENGINE(std::bfloat16_t)
#endif
