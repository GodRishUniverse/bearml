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



   template<typename T> class  Node;
   // TODO: fix for MATRIX AND TENSOR Types - probably will need to provide template specializations
   // TODO: add double and Tensor operator overloads to unblock the loss functions like log loss - by implementing operator overloads
   template <typename T>
   class Node {
    public:
        T val;
        T grad; // delay grad creation and use jacobians to solbe the broadcasting problem
        std::vector<std::shared_ptr<Node<T>>> inputs;// PARENTS-  using stl shared pointer
        std::vector<std::weak_ptr<Node<T>>> outputs; // CHILDREN- using stl weak pointer - to break the cycle of shared_ptr references in inputs and outputs
        std::function<void()> backward_fn; // will be used for backward pass rather than the gradients

        Node(T value) : val(value) , grad([&value]() {
                if constexpr (std::is_same<T, double>::value) {
                    return 0.0;
                } else if constexpr (std::is_same<T, simplenet::Tensor>::value) {
                    return simplenet::Tensor(value.getShape());
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

        void backward(){
            // if we have this defined then we will call it
            if (backward_fn){
                backward_fn();
            }
        }

        friend std::shared_ptr<Node<T>> operator+(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
                std::shared_ptr<Node<T>> node  = make_node(a->val+b->val);
                // node->grad = a->grad + b->grad;  // c = a + b     =>    dc = da + db

                node->inputs = {a,b};

                a->outputs.push_back(node);
                b->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_b = b;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a, weak_b, weak_node](){

                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    if (!a_locked || !b_locked || !node_locked) {
                        return; // One of the nodes was destroyed
                    }
                    if constexpr (std::is_same<T, simplenet::Tensor>::value){
                        std::vector<int> temp_a =  a_locked->grad.getShape();
                        std::vector<int> temp_b = b_locked->grad.getShape();
                        a_locked->grad += simplenet::linear_algebra::reduce(node_locked->grad, temp_a);  // dc/da = b
                        b_locked->grad += simplenet::linear_algebra::reduce(node_locked->grad, temp_b);  // dc/db = a
                    }else{
                        // case for doubles
                        a_locked->grad += node_locked->grad;  // dc/da = b
                        b_locked->grad += node_locked->grad;  // dc/db = a
                    }

                };

                return node;
        }

        friend std::shared_ptr<Node<T>> operator+(double scalar, std::shared_ptr<Node<T>> b){
            simplenet::Tensor scalar_tensor({1});
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return scalar_node + b;
        }

        friend std::shared_ptr<Node<T>> operator+( std::shared_ptr<Node<T>> b, double scalar){
            simplenet::Tensor scalar_tensor({1});
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return b + scalar_node;
        }


        friend std::shared_ptr<Node<T>> operator-(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
                std::shared_ptr<Node<T>> node  = make_node(a->val-b->val);
                // node->grad = a->grad - b->grad;  // c = a - b     =>    dc = da - db

                node->inputs = {a,b};

                a->outputs.push_back(node);
                b->outputs.push_back(node);


                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_b = b;
                std::weak_ptr<Node<T>> weak_node = node;


                node->backward_fn = [weak_a, weak_b, weak_node](){
                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    if (!a_locked || !b_locked || !node_locked) {
                        return; // One of the nodes was destroyed
                    }

                    if constexpr (std::is_same<T, simplenet::Tensor>::value){
                        std::vector<int> temp_a =  a_locked->grad.getShape();
                        std::vector<int> temp_b = b_locked->grad.getShape();

                        a_locked->grad += simplenet::linear_algebra::reduce(node_locked->grad,temp_a);  // dc/da = 1
                        b_locked->grad += simplenet::linear_algebra::reduce(node_locked->grad * -1.0,temp_b);  // dc/db = -1
                    }else{
                        // case for doubles
                        a_locked->grad += node_locked->grad*1.0;  // dc/da = 1
                        b_locked->grad += node_locked->grad*-1.0;  // dc/db = -1
                    }
                };

                return node;
        }

        friend std::shared_ptr<Node<T>> operator-(double scalar, std::shared_ptr<Node<T>> b){
            simplenet::Tensor scalar_tensor({1});
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return scalar_node - b;
        }

        friend std::shared_ptr<Node<T>> operator-( std::shared_ptr<Node<T>> b, double scalar){
            simplenet::Tensor scalar_tensor({1});
            scalar_tensor.fill(scalar);

            auto scalar_node = constant(scalar_tensor);
            return b - scalar_node;
        }


        friend std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){

                std::shared_ptr<Node<T>> node  = make_node(a->val*b->val);
                // node->grad = b->val*a->grad + b->grad*a->val;  //  dc = a * b     =>    dc = b * da + a * db

                node->inputs = {a,b};

                a->outputs.push_back(node);
                b->outputs.push_back(node);


                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_b = b;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a, weak_b, weak_node](){

                    auto a_locked = weak_a.lock();
                    auto b_locked = weak_b.lock();
                    auto node_locked = weak_node.lock();

                    if (!a_locked || !b_locked || !node_locked) {
                        return; // One of the nodes was destroyed
                    }

                    if constexpr (std::is_same<T, simplenet::Tensor>::value){
                        std::vector<int> temp_a =  a_locked->grad.getShape();
                        std::vector<int> temp_b = b_locked->grad.getShape();
                        a_locked->grad += simplenet::linear_algebra::reduce(node_locked->grad * b_locked->val.transpose(),temp_a); // grad_a = grad * b^T
                        b_locked->grad += simplenet::linear_algebra::reduce(a_locked->val.transpose() * node_locked->grad, temp_b); // grad_b = a^T * grad
                    }else{
                        // case for doubles
                        a_locked->grad += node_locked->grad * b_locked->val; // grad_a = grad * b^T
                        b_locked->grad += a_locked->val * node_locked->grad; // grad_b = a^T * grad
                    }
                };
                return node;
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
                        b_locked->grad += simplenet::linear_algebra::reduce(scalar * node_locked->grad, temp_b); // grad_b = scalar * grad
                    }else{
                       // TODO
                    }
                };
                return node;
        }

        friend std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> b, double scalar){
               return scalar * b; // functionality is the same - doesnt matter if the scalar is on the left or right
        }


        // unary operator: e^x
        friend std::shared_ptr<Node<T>> exponent(std::shared_ptr<Node<T>> a){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){
                // std::cout <<"EXPONENTIATED TENSOR NODE" <<std::endl;

                std::shared_ptr<Node<T>> node  = make_node(simplenet::Tensor::exp(a->val));
                //  c = exp(a)     =>    dc/da = exp(a)

                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a, weak_node]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();

                    if(!a_locked || !node_locked){
                        return; // One of the nodes was destroyed
                    }
                    std::vector<int> a_shape = a_locked->grad.getShape();
                    a_locked->grad +=  simplenet::linear_algebra::reduce(node_locked->grad * node_locked->val, a_shape);  //  dc/da = exp(a)
                };

                return node;
            }else{
                std::cout << "TYPE HERE IS - > " << typeid(T).name() << std:: endl; // debugging snippet
                std::shared_ptr<Node<T>> node  = make_node(exp(a->val)); // TODO: implement exp for tensors and matrices
                //  c = exp(a)     =>    dc/da = exp(a)

                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;


                node->backward_fn = [weak_a, weak_node]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();

                    if(!a_locked || !node_locked){
                        return; // One of the nodes was destroyed
                    }
                    a_locked->grad += node_locked->grad * node_locked->val;  //  dc/da = exp(a)
                };

                return node;
            }

        }

        // unary operator: transpose
        friend std::shared_ptr<Node<T>> transpose(std::shared_ptr<Node<T>> a){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){
                std::shared_ptr<Node<T>> node  = make_node(a->val.transpose());

                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;


                node->backward_fn = [weak_a, weak_node]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();
                    a_locked->grad +=  node_locked->grad.transpose();
                };

                return node;
            }

            // we dont do anything for the doubles
            return a; // unchanged
        }

        // TODO: test and also add ops for when the max is double values
        friend std::shared_ptr<Node<T>> max(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
            // SHAPES HAVE TO BE THE SAME
            if (a->val.getShape() != b->val.getShape()){
                throw std::invalid_argument("Shapes incompatible for max operation to be backpropagated");
            }
            if constexpr (std::is_same<T, simplenet::Tensor>::value){

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
            // SHAPES HAVE TO BE THE SAME
            if (a->val.getShape() != b->val.getShape()){
                throw std::invalid_argument("Shapes incompatible for min operation to be backpropagated");
            }
            if constexpr (std::is_same<T, simplenet::Tensor>::value){

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
            // SHAPES HAVE TO BE THE SAME
            if (a->val.getShape() != b->val.getShape()){
                throw std::invalid_argument("Shapes incompatible for hadamard operation to be backpropagated");
            }
            if constexpr (std::is_same<T, simplenet::Tensor>::value){

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



        // unary operator
        // d|f(x)|/dx = sign(f(x))
        friend std::shared_ptr<Node<T>> abs(std::shared_ptr<Node<T>> a){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){
                std::shared_ptr<Node<T>> node  = make_node(simplenet::Tensor::abs(a->val));
                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a , weak_node]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();
                    simplenet::Tensor sign_mat = simplenet::linear_algebra::sign(a_locked->val);
                    a_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, sign_mat);
                };

                return node;
            }else{
              // To Implement
            }
        }


        static std::shared_ptr<Node<T>> log(std::shared_ptr<Node<T>> a){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){
                std::shared_ptr<Node<T>> node  = make_node(simplenet::Tensor::log(a->val));
                node->inputs = {a};
                a->outputs.push_back(node);

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a , weak_node]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();
                    // d/da log(a) = 1/a
                    a_locked->grad +=  simplenet::linear_algebra::hadamard(node_locked->grad, 1.0/a_locked->val);
                };

                return node;
            }else{
              // To Implement
            }
        }


        // unary operator
        friend std::shared_ptr<Node<T>> mean(std::shared_ptr<Node<T>> a){
            if constexpr (std::is_same<T, simplenet::Tensor>::value){
                std::shared_ptr<Node<T>> node  = make_node(simplenet::Tensor::mean(a->val));
                node->inputs = {a};
                a->outputs.push_back(node);

                // c = mean(a)
                // c = 1/n * sum(a)
                // dL/da = dL/dc * dc/da (sum has gradient as 1 so only 1/n remains)

                std::weak_ptr<Node<T>> weak_a = a;
                std::weak_ptr<Node<T>> weak_node = node;

                node->backward_fn = [weak_a , weak_node]() {
                    auto a_locked = weak_a.lock();
                    auto node_locked = weak_node.lock();
                    ll n = a_locked->val.sizeOfTensor();
                    simplenet::Tensor grad_broadcast(a_locked->val.getShape());
                    double scalar_grad = node_locked->grad.get({0});
                    grad_broadcast.fill(scalar_grad / static_cast<double>(n));

                    a_locked->grad+= grad_broadcast;
                };

                return node;
            }else{
              // To Implement
            }
        }


        // DIVISION is confusing - should only be done for numbers
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
