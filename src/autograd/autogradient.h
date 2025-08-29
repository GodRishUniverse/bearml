#pragma once

#include <algorithm>
#include <typeinfo>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <type_traits>


#include "tensor/Tensor.h"
#include "matrix/Matrix.h"
#include "activation_functions.cpp"

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


    // TODO: implement the computational graph elements - binary and unary operations

   template<typename T> class  Node;
   // TODO: fix for MATRIX AND TENSOR Types - probably will need to provide template specializations
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

        void backward(){
            // if we have this defined then we will call it
            if (backward_fn){
                backward_fn();
            }
        }

        // TODO: need to think through how to apply the reduce operation for the multiplication

        // TODO - change
        friend std::shared_ptr<Node<T>> operator+(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
                std::shared_ptr<Node<T>> node  = make_node(a->val+b->val);
                // node->grad = a->grad + b->grad;  // c = a + b     =>    dc = da + db

                node->inputs = {a,b};

                a->outputs.push_back(node);
                b->outputs.push_back(node);

                // TODO - change this function as not correct with Tensors -  shape changes and broadcasting has a role to play here - if forward does broadcasting then backward should do reduce
                node->backward_fn = [a, b, node](){
                    a->grad += node->grad;  // dc/da = b
                    b->grad += node->grad;  // dc/db = a
                };

                return node;
        }


        friend std::shared_ptr<Node<T>> operator-(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){
                std::shared_ptr<Node<T>> node  = make_node(a->val-b->val);
                // node->grad = a->grad - b->grad;  // c = a - b     =>    dc = da - db

                node->inputs = {a,b};

                a->outputs.push_back(node);
                b->outputs.push_back(node);

                // TODO - change this function as not correct with Tensors -  shape changes and broadcasting has a role to play here
                node->backward_fn = [a, b, node](){
                    a->grad += node->grad*1.0;  // dc/da = 1
                    b->grad += node->grad*-1.0;  // dc/db = -1
                };


                return node;
        }

        friend std::shared_ptr<Node<T>> operator*(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b){

                std::shared_ptr<Node<T>> node  = make_node(a->val*b->val);
                // node->grad = b->val*a->grad + b->grad*a->val;  //  dc = a * b     =>    dc = b * da + a * db

                node->inputs = {a,b};

                // TODO - change this function as not correct with Tensors - shape changes and broadcasting has a role to play here
                a->outputs.push_back(node);
                b->outputs.push_back(node);

                node->backward_fn = [a, b, node](){
                    a->grad +=  node->grad * b->val;  // dc/da = b
                    b->grad +=  node->grad * a->val; // dc/db = a
                };


                return node;
        }


        // DIVISION is confusing - should only be done for numbers
        // specialized for constants
        // TODO: change for Tensors
        friend std::shared_ptr<Node<T>> operator/(std::shared_ptr<Node<T>> a, double divisor){
                std::shared_ptr<Node<T>> node  = make_node(a->val/divisor); // TODO: implement Matrix and Tensor division for constant values
                // node->grad = (a->grad*divisor)/ (divisor * divisor); //  c = a / b     =>    dc = (da *b  - a*db)/b^2

                node->inputs = {a};
                a->outputs.push_back(node);

                // TODO - change this function as not correct with Tensors - shape changes and broadcasting has a role to play here
                node->backward_fn = [a, divisor, node]() {
                    a->grad += node->grad * (1.0 / divisor);  // dc/da = 1/divisor
                };


                return node;
        }

        // unary operator: e^x
        friend std::shared_ptr<Node<T>> exp(std::shared_ptr<Node<T>> a){
            //TODO: will need to use template specialization for Tensors and Matrices
            std::shared_ptr<Node<T>> node  = make_node(exp(a->val)); // TODO: implement exp for tensors and matrices
            //  c = exp(a)     =>    dc/da = exp(a)

            node->inputs = {a};
            a->outputs.push_back(node);

            node->backward_fn = [a, node]() {
                a->grad += node->grad * exp(a->val);  //  dc/da = exp(a)
            };

            return node;
        }

    };

    namespace autogradient {
        template <typename T>
        T backward(std::shared_ptr<simplenet::Node<T>> end_node, bool accumulate = false);
    }


   // Edges will be operation types on two different Nodes (also same node as well)
   // - need to think how gradients will change with some different operations like Permute
    // ANSWER to above question is:
    // Permuting a tensor changes its shape but doesn't affect the underlying data or its relationships to
    // other tensors in the computation graph. Consequently, the gradient of a permuted tensor is simply the permutation of the original gradient,
    // with the same operations applied to the gradient's axes as were applied to the tensor's axes.


}
#endif
