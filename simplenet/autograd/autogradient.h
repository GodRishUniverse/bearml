#pragma once

#include <algorithm>
#include <stdexcept>
#include <typeinfo>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <type_traits>
#include <iostream>


#include "tensor/Tensor.h"
// #include "../cuda/includes/ops.h" //TODO: this Ops should be moved to a separate folder as Ops.h used in Tensor.h as well


using ll = long long; // can also use int_fast64_t

#ifndef AUTO_NODE_H
#define AUTO_NODE_H
// Will use reverse mode -  https://en.wikipedia.org/wiki/Automatic_differentiation
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
namespace simplenet{

    // Notes for implementation
    /*

    We will be using a topological sort to walk the graph and compute the derivatives
    We will design a computational graph and use a topological sort


    https://en.wikipedia.org/wiki/Topological_sort

    */


    // We need to create a representation for a computation graph
    // then we can use a reverse topological order to walk the graph and compute the derivatives backward from the output
    //    x1 ---- x3 ---- out
    //    x2-----/
    //      here we have x1, x2, x3, out - we start the backward pass from out to compute the derivatives of out with respect to all the nodes
    // Algorithm is as follows:

    /*

    def gradient(out):
        node_to_grad = {out: [1]} // dictionary for node to list of gradients for each node that is connected to it

        for i in reverse_topological_sort(out):
            v_i_adjoint = sum(node_to_grad[i])

            for k in inputs(i):
                compute v_k_to_i_adjoint = v_i_adjoint * d_v_i / d_v_k
                append v_k_to_i_adjoint to node_to_grad[k] // propagate the partial adjoints to its inputs
        return adjoint of input v_input(s) - this is the gradient of the function (network) with respect to all the inputs
    */


   // TODO refactor with ops
   template<typename T> class  Node;
   // TODO: fix for MATRIX AND TENSOR Types - probably will need to provide template specializations
   // TODO: add double and Tensor operator overloads to unblock the loss functions like log loss - by implementing operator overloads
   template <typename T>
   class Node {
        // C++17 inline static variable - only one copy shared across all instances of the class
        inline static simplenet::reductions::ReductionOps reduction_op = simplenet::reductions::ReductionOps::SUM;
    public:
        T val;
        T grad; // delay grad creation and use jacobians to solbe the broadcasting problem
        std::vector<std::shared_ptr<Node<T>>> inputs;// PARENTS-  using stl shared pointer
        std::vector<std::weak_ptr<Node<T>>> outputs; // CHILDREN- using stl weak pointer - to break the cycle of shared_ptr references in inputs and outputs
        std::function<void()> backward_fn; // will be used for backward pass rather than the gradients


        Node(T value ) : val(value) , grad([&value]() {
                if constexpr (std::is_same<T, double>::value) {
                    return 0.0;
                } else if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                    return simplenet::Tensor(value.getShape(), value.getDevice()  );
                }
            }()) {
        }

        // Factory functions that return shared_ptr
        static std::shared_ptr<Node<T>> make_node(T value) {
            return std::make_shared<Node<T>>(value);
        }

        // a utility function for creating constant nodes
        static std::shared_ptr<Node<T>> constant(T value) {
            auto node = make_node(value);
            return node; // this is a leaf node with no inputs and no backward_fn
        }

        // a utility to set the reduction operation for all nodes
        static void set_reduction(simplenet::reductions::ReductionOps op) {
            reduction_op = op;
        }

        // a utility to get the current reduction operation
        static simplenet::reductions::ReductionOps get_reduction() {
            return reduction_op;
        }

        void backward(){
            // if we have this defined then we will call it
            if (backward_fn){
                backward_fn();
            }
        }

        static void accumulate_grad(T& target_grad, const T& grad_contribution, const T& target_val) {
            if constexpr (std::is_same_v<T, simplenet::Tensor>) {
                auto target_grad_shape = target_val.getShape();
                target_grad += simplenet::linear_algebra::reduce(grad_contribution, target_grad_shape, reduction_op);
            } else {
                target_grad += grad_contribution;
            }
        }

        // refactoring
        friend std::shared_ptr<Node<T>> elementwise_binary_node_operators(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b, OP_Code op){
            std::shared_ptr<Node<T>> node;
            switch(op) {
                case OP_Code::OP_ADD: node = make_node(a->val + b->val); break;
                // node->grad = a->grad + b->grad;  // c = a + b     =>    dc = da + db
                case OP_Code::OP_SUB: node = make_node(a->val - b->val); break;
                // node->grad = a->grad - b->grad;  // c = a - b     =>    dc = da - db
                case OP_Code::OP_MUL: node  = make_node(a->val*b->val); break;
                // node->grad = b->val*a->grad + b->grad*a->val;  //  dc = a * b     =>    dc = b * da + a * db

                default:
                    throw std::invalid_argument("Autograd: OP Code does not exist - in elementwise_binary.");
            }

            node->inputs = {a,b};

            a->outputs.push_back(node);
            b->outputs.push_back(node);

            std::weak_ptr<Node<T>> weak_a = a;
            std::weak_ptr<Node<T>> weak_b = b;
            std::weak_ptr<Node<T>> weak_node = node;

            node->backward_fn = [weak_a, weak_b, weak_node, op](){

                auto a_locked = weak_a.lock();
                auto b_locked = weak_b.lock();
                auto node_locked = weak_node.lock();

                if (!a_locked || !b_locked || !node_locked) {
                    return; // One of the nodes was destroyed
                }

                switch (op) {
                    case OP_Code::OP_ADD:
                        accumulate_grad(a_locked->grad, node_locked->grad, a_locked->val); // dc/da = 1
                        accumulate_grad(b_locked->grad, node_locked->grad, b_locked->val); // dc/db = 1
                        break;
                    case OP_Code::OP_SUB:
                        accumulate_grad(a_locked->grad, node_locked->grad, a_locked->val); // dc/da = 1
                        accumulate_grad(b_locked->grad, node_locked->grad*-1.0, b_locked->val); // dc/db = -1
                        break;
                    case OP_Code::OP_MUL:
                        if constexpr (std::is_same<T, simplenet::Tensor>::value){
                            accumulate_grad(a_locked->grad, node_locked->grad * b_locked->val.transpose(), a_locked->val); // grad_a = grad * b^T
                            accumulate_grad(b_locked->grad,a_locked->val.transpose() * node_locked->grad, b_locked->val); // grad_b = a^T * grad
                        }else {
                            // doubles (no transpose needed for scalars)
                            a_locked->grad += node_locked->grad * b_locked->val; // grad_a = grad * b^T
                            b_locked->grad += a_locked->val * node_locked->grad; // grad_b = a^T * grad
                        }
                        break;
                    default:
                        throw std::invalid_argument("Autograd: OP Code does not exist - in elementwise_binary.");
                }

            };

            return node;
        }

        // Our primary elementwise unary node operator function (called in every unary node operator)
        friend std::shared_ptr<Node<T>> elementwise_unary_node_operators(std::shared_ptr<Node<T>> a, OP_Code op){
            std::shared_ptr<Node<T>> node;

            // std::cerr << "DEBUGGING: TYPE HERE IS - > " << typeid(T).name() << std:: endl; // debugging snippet

            switch(op) {
                case OP_Code::OP_EXP:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::exp(a->val));
                    else node = make_node(exp(a->val));
                    break;
                case OP_Code::OP_SIN:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::sin(a->val));
                    else node = make_node(sin(a->val));
                    break;
                case OP_Code::OP_COS:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::cos(a->val));
                    else node = make_node(cos(a->val));
                    break;
                case OP_Code::OP_TAN:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::tan(a->val));
                    else node = make_node(tan(a->val));
                    break;
                case OP_Code::OP_SINH:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::sinh(a->val));
                    else node = make_node(sinh(a->val));
                    break;
                case OP_Code::OP_COSH:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::cosh(a->val));
                    else node = make_node(cosh(a->val));
                    break;
                case OP_Code::OP_TANH:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::tanh(a->val));
                    else node = make_node(tanh(a->val));
                    break;
                case OP_Code::OP_TRANSPOSE:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(a->val.transpose());
                    else node = make_node(a->val); // same as input for non-Tensor types
                    break;
                case OP_Code::OP_ABS:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)   node =  make_node(simplenet::Tensor::abs(a->val));
                    else node = make_node(abs(a->val));
                    break;
                case OP_Code::OP_LOG:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)  node  = make_node(simplenet::Tensor::log(a->val));
                    else node = make_node(std::log(a->val));
                    break;
                case OP_Code::OP_MEAN_FOR_GRAD:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)  node  = make_node(simplenet::Tensor::mean(a->val));
                    else node = make_node(a->val);
                    break;
                case OP_Code::OP_SQRT:
                    if constexpr (std::is_same<T, simplenet::Tensor>::value)  node  = make_node(simplenet::Tensor::sqrt(a->val));
                    else node = make_node(std::sqrt(a->val));
                    break;
                default:
                    throw std::invalid_argument("Autograd: OP Code does not exist - in elementwise_unary.");
            }

            node->inputs = {a};
            a->outputs.push_back(node);

            std::weak_ptr<Node<T>> weak_a = a;
            std::weak_ptr<Node<T>> weak_node = node;

            node->backward_fn = [weak_a, weak_node, op]() {
                auto a_locked = weak_a.lock();
                auto node_locked = weak_node.lock();

                if(!a_locked || !node_locked){
                    return; // One of the nodes was destroyed
                }
                switch(op) {
                    case OP_Code::OP_EXP:
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, node_locked->val), a_locked->grad);  //  dc/da = exp(a)
                        } else {
                            a_locked->grad += node_locked->grad * node_locked->val;  //  dc/da = exp(a)
                        }
                        break;
                    case OP_Code::OP_SIN:
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, simplenet::Tensor::cos(a_locked->val)), a_locked->grad);  //  dc/da = cos(a)
                        } else {
                            a_locked->grad += node_locked->grad * cos(node_locked->val);  //  dc/da = cos(a)
                        }
                        break;
                    case OP_Code::OP_COS:
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, (-1.0 *  simplenet::Tensor::sin(a_locked->val))), a_locked->grad);  //  dc/da = -sin(a)
                        } else {
                            a_locked->grad += node_locked->grad * T(-1.0) * sin(node_locked->val);  //  dc/da = -sin(a)
                        }
                        break;
                    case OP_Code::OP_TAN:
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            auto tan_val = simplenet::Tensor::tan(a_locked->val);
                            //  dc/da = sec^2(a) = 1 + tan^2(a)
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, (1.0 + simplenet::linear_algebra::hadamard(tan_val, tan_val)))), a_locked->grad);
                        } else {
                            auto tan_val = tan(a_locked->val);
                            a_locked->grad += node_locked->grad * (T(1.0) + tan_val * tan_val);  //  dc/da = sec^2(a) = 1 + tan^2(a)
                        }
                        break;
                    case OP_Code::OP_SINH:
                        if constexpr (std::is_same<T,simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, simplenet::Tensor::cosh(a_locked->val)), a_locked->grad); //  dc/da = cosh(a)
                        } else {
                            a_locked->grad += node_locked->grad *  cosh(a_locked->val);;  //  dc/da = cosh(a)
                        }
                        break;
                    case OP_Code::OP_COSH:
                        if constexpr (std::is_same<T,simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, simplenet::Tensor::sinh(a_locked->val)), a_locked->grad); //  dc/da = sinh(a)
                        } else {
                            a_locked->grad += node_locked->grad * sinh(a_locked->val);  //  dc/da = sinh(a)
                        }
                        break;
                    case OP_Code::OP_TANH:
                        if constexpr (std::is_same<T,simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, (1.0 - simplenet::linear_algebra::hadamard(node_locked->val, node_locked->val))), a_locked->grad); //  dc/da = 1 - tanh^2(a)
                        } else {
                            a_locked->grad += node_locked->grad * (T(1.0) - node_locked->val * node_locked->val);  //  dc/da = 1 - tanh^2(a)
                        }
                        break;
                    case OP_Code::OP_TRANSPOSE:
                        // here reduce will not be applied - shapes are the same (handled by accumulate_grad)
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, node_locked->grad.transpose(), a_locked->val);
                        }
                        break;
                    case OP_Code::OP_ABS:
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            simplenet::Tensor sign_mat = simplenet::linear_algebra::sign(a_locked->val);
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, sign_mat), a_locked->val);
                        } else {
                            a_locked->grad += node_locked->grad * (std::signbit(a_locked->val) ? T(-1.0) : (a_locked->val == T(0.0) ? T(0.0) : T(1.0)));
                        }
                        break;
                    case OP_Code::OP_LOG:
                        // d/da log(a) = 1/a
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, 1.0/a_locked->val), a_locked->val);
                        } else {
                            a_locked->grad += node_locked->grad * (T(1.0) / a_locked->val);
                        }
                        break;
                    case OP_Code::OP_MEAN_FOR_GRAD:
                        // c = mean(a)
                        // c = 1/n * sum(a)
                        // dL/da = dL/dc * dc/da (sum has gradient as 1 so only 1/n remains)
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            ll n = a_locked->val.sizeOfTensor();
                            simplenet::Tensor grad_broadcast(a_locked->val.getShape(), a_locked->val.getDevice());
                            simplenet::Tensor grad_cpu = node_locked->grad.to(simplenet::Device::cpu());
                            double scalar_grad = grad_cpu.get({0}); // we shift to the cpu because GPU direct access is not supported
                            grad_broadcast.fill(scalar_grad / static_cast<double>(n));
                            accumulate_grad(a_locked->grad, grad_broadcast, a_locked->val);
                        }else{
                            a_locked->grad += node_locked->grad;
                        }
                       break;
                    case OP_Code::OP_SQRT:
                        // c = sqrt(a)
                        // dc/da = 1/(2*sqrt(a))
                        if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                            accumulate_grad(a_locked->grad, simplenet::linear_algebra::hadamard(node_locked->grad, 1.0 / (2.0 * simplenet::Tensor::sqrt(a_locked->val))), a_locked->val);
                        }else{
                            a_locked->grad += node_locked->grad / (2.0 * std::sqrt(a_locked->val));
                        }
                        break;
                    default:
                        throw std::invalid_argument("Autograd: OP Code does not exist - in elementwise_unary.");
                }

            };
            return node;
        }

        friend std::shared_ptr<Node<T>> operator+(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
               return elementwise_binary_node_operators(a, b, OP_Code::OP_ADD);
        }

        friend std::shared_ptr<Node<T>> operator+(double scalar, std::shared_ptr<Node<T>> b){
            simplenet::Tensor scalar_tensor({1}, b->val.getDevice());
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return scalar_node + b;
        }

        friend std::shared_ptr<Node<T>> operator+( std::shared_ptr<Node<T>> b, double scalar){
            simplenet::Tensor scalar_tensor({1}, b->val.getDevice());
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return b + scalar_node;
        }


        friend std::shared_ptr<Node<T>> operator-(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
            return elementwise_binary_node_operators(a, b, OP_Code::OP_SUB);
        }

        friend std::shared_ptr<Node<T>> operator-(double scalar, std::shared_ptr<Node<T>> b){
            simplenet::Tensor scalar_tensor({1},  b->val.getDevice());
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return scalar_node - b;
        }

        friend std::shared_ptr<Node<T>> operator-( std::shared_ptr<Node<T>> b, double scalar){
            simplenet::Tensor scalar_tensor({1},  b->val.getDevice());
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return b - scalar_node;
        }


        friend std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
            return elementwise_binary_node_operators(a, b, OP_Code::OP_MUL);
        }

        friend std::shared_ptr<Node<T>> operator*(double scalar, std::shared_ptr<Node<T>> b){

                std::shared_ptr<Node<T>> node  = make_node(scalar*b->val);
                // node->grad = b->val*a->grad + b->grad*a->val;  //  dc = a * b     =>    dc = b * da + a * db

                node->inputs = {b};

                b->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_b = b;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [scalar, weak_b, weak_node](){

                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    if (!b_locked || !node_locked) {
                        return; // One of the nodes was destroyed
                    }

                    if constexpr (std::is_same<T, simplenet::Tensor>::value){
                        std::vector<int> temp_b = b_locked->grad.getShape();
                        b_locked->grad += simplenet::linear_algebra::reduce(scalar * node_locked->grad, temp_b, reduction_op); // grad_b = scalar * grad
                    }else{
                       // TODO
                    }
                };
                return node;
        }

        friend std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> b, double scalar){
               return scalar * b; // functionality is the same - doesnt matter if the scalar is on the left or right
        }

        // TODO: test and also add ops for when the max is double values
        friend std::shared_ptr<Node<T>> max(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){

                // SHAPES HAVE TO BE THE SAME
                if (a->val.getShape() != b->val.getShape()){
                    throw std::invalid_argument("Shapes incompatible for max operation to be backpropagated");
                }

                std::shared_ptr<Node<T>> node  = make_node(simplenet::Tensor::max(a->val, b->val));
                node->inputs = {a,b};
                a->outputs.push_back(node);
                b->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;
                std::weak_ptr<Node<T>> weak_b = b;

                node->backward_fn = [weak_a ,weak_b, weak_node]() {

                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    // Create masks for a and b
                    // like ->
                    //  Tensor a_mask = (a->val >= b->val).cast<double>(); // 1 where a >= b, 0 elsewhere
                    //  Tensor b_mask = (b->val >= a->val).cast<double>();  // 1 where b >= a, 0 elsewhere
                    simplenet::Tensor a_mask = simplenet::linear_algebra::mask_of_greater_than_equal_to(a_locked->val,b_locked->val);
                    simplenet::Tensor b_mask = simplenet::linear_algebra::mask_of_greater_than_equal_to(b_locked->val,a_locked->val);

                    // Need to figure out how the multiplication of masks will be done - most probably hadamard
                    a_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, a_mask);
                    b_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, b_mask);

                };

                return node;
            }else{
              // To Implement
            }

        }


        // TODO: test and also add ops for when the min is double values
        friend std::shared_ptr<Node<T>> min(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){

                // SHAPES HAVE TO BE THE SAME
                if (a->val.getShape() != b->val.getShape()){
                    throw std::invalid_argument("Shapes incompatible for min operation to be backpropagated");
                }

                std::shared_ptr<Node<T>> node  = make_node(simplenet::Tensor::min(a->val, b->val));
                node->inputs = {a,b};
                a->outputs.push_back(node);
                b->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;
                std::weak_ptr<Node<T>> weak_b = b;

                node->backward_fn = [weak_a ,weak_b, weak_node]() {

                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    // Create masks for a and b
                    // like ->
                    //  Tensor a_mask = (a->val <= b->val).cast<double>(); // 1 where a <= b, 0 elsewhere
                    //  Tensor b_mask = (b->val <= a->val).cast<double>();  // 1 where b <= a, 0 elsewhere
                    simplenet::Tensor a_mask = simplenet::linear_algebra::mask_of_less_than_equal_to(a_locked->val,b_locked->val);
                    simplenet::Tensor b_mask = simplenet::linear_algebra::mask_of_less_than_equal_to(b_locked->val,a_locked->val);

                    // Need to figure out how the multiplication of masks will be done - most probably hadamard
                    a_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, a_mask);
                    b_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, b_mask);

                };

                return node;
            }else{
              // To Implement
            }
        }

        // hadamard product differentiation is just product rule -
        // d(a*b)/dx = a * db/dx + b * da/dx
        static std::shared_ptr<Node<T>> hadamard(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){

            if constexpr (std::is_same<T, simplenet::Tensor>::value){

                // SHAPES HAVE TO BE THE SAME
                if (a->val.getShape() != b->val.getShape()){
                    throw std::invalid_argument("Shapes incompatible for hadamard operation to be backpropagated");
                }

                std::shared_ptr<Node<T>> node  = make_node(simplenet::linear_algebra::hadamard(a->val, b->val));
                node->inputs = {a,b};
                a->outputs.push_back(node);
                b->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;
                std::weak_ptr<Node<T>> weak_b = b;

                node->backward_fn = [weak_a ,weak_b, weak_node]() {

                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    a_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, b_locked->val);
                    b_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, a_locked->val);

                };

                return node;
            }else{
              // To Implement
            }
        }

        // DIVISION uses hadamard product
        friend std::shared_ptr<Node<T>> operator/(std::shared_ptr<Node<T>> a, double divisor){
                std::shared_ptr<Node<T>> node  = make_node(a->val/divisor);
                // node = a / b
                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a, weak_node,  divisor]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();
                    a_locked->grad += node_locked->grad / divisor; // dL/da = dL/dc × dc/da
                };
                return node;
        }

        friend std::shared_ptr<Node<T>> operator/(double divisor, std::shared_ptr<Node<T>> a){
                std::shared_ptr<Node<T>> node  = make_node(divisor/a->val);
                // node = a / b
                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a, weak_node,  divisor]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();

                    // dc/da = grad * (-divisor / (a^2))
                    if constexpr (std::is_same<T, simplenet::Tensor>::value){
                        a_locked->grad += simplenet::linear_algebra::hadamard(node_locked->grad,
                            -divisor / simplenet::linear_algebra::hadamard(a_locked->val, a_locked->val));
                    } else {
                        a_locked->grad += node_locked->grad * (-divisor / (a_locked->val * a_locked->val));
                    }

                };
                return node;
        }

        friend std::shared_ptr<Node<T>> operator/(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){

            if constexpr (std::is_same<T, simplenet::Tensor>::value){

                // SHAPES HAVE TO BE THE SAME - hadamard operation
                if (a->val.getShape() != b->val.getShape()){
                    throw std::invalid_argument("Shapes incompatible for hadamard operation to be backpropagated");
                }

                std::shared_ptr<Node<T>> node  = make_node(a->val/ b->val);
                node->inputs = {a,b};
                a->outputs.push_back(node);
                b->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;
                std::weak_ptr<Node<T>> weak_b = b;

                node->backward_fn = [weak_a ,weak_b, weak_node]() {

                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();
                    // c = a/b
                    // dc/da = grad * (1/b)
                    // dc/db = grad * (-a/b^2)
                    a_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, 1.0/b_locked->val);
                    b_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, -1.0 * a_locked->val/ simplenet::linear_algebra::hadamard(b_locked->val, b_locked->val));

                };

                return node;
            }else{
              // To Implement
            }
        };

        // UNARY Operators:

        // unary operator: e^x
        friend std::shared_ptr<Node<T>> exp(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_EXP);
        }


        // unary operator: sin(x)
        friend std::shared_ptr<Node<T>> sin(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_SIN);
        }


        // unary operator: cos(x)
        friend std::shared_ptr<Node<T>> cos(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_COS);
        }

        // unary operator: tan(x)
        friend std::shared_ptr<Node<T>> tan(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_TAN);
        }


        // unary operator: sinh(x)
        friend std::shared_ptr<Node<T>> sinh(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_SINH);
        }


        // unary operator: cosh(x)
        friend std::shared_ptr<Node<T>> cosh(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_COSH);
        }

        // unary operator: tanh(x)
        friend std::shared_ptr<Node<T>> tanh(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_TANH);
        }



        // unary operator: transpose
        friend std::shared_ptr<Node<T>> transpose(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_TRANSPOSE);
        }



        // unary operator
        // d|f(x)|/dx = sign(f(x))
        friend std::shared_ptr<Node<T>> abs(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_ABS);
        }


        static std::shared_ptr<Node<T>> log(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_LOG);
        }


        // unary operator
        friend std::shared_ptr<Node<T>> mean(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_MEAN_FOR_GRAD);
        }

        // unary operator
        friend std::shared_ptr<Node<T>> sqrt(std::shared_ptr<Node<T>> a){
            return elementwise_unary_node_operators(a, OP_Code::OP_SQRT);
        }

    };




    namespace autogradient {
        template <typename T>
        T backward(std::shared_ptr<simplenet::Node<T>> end_node, bool accumulate = false);
    }

}
#endif




// OLD NOTES
// Edges will be operation types on two different Nodes (also same node as well)
// - need to think how gradients will change with some different operations like Permute
// ANSWER to above question is:
// Permuting a tensor changes its shape but doesn't affect the underlying data or its relationships to
// other tensors in the computation graph. Consequently, the gradient of a permuted tensor is simply the permutation of the original gradient,
// with the same operations applied to the gradient's axes as were applied to the tensor's axes.
