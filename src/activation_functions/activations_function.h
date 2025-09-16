#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"

using ll = long long;

#ifndef ACTIVATION_FUNCTIONS
namespace simplenet {

    namespace neural_network {
        // will apply polymorphism and inheritance for activation functions as the principle idea is the same
        // will inherit the Node class for operations to be carried out and use the backward features

        class Module{
            public:
                virtual ~Module()  = default; // it is a pure virtual class

                // we will assume at least one input - may change it
                virtual simplenet::Node<Tensor> forward(std::shared_ptr<simplenet::Node<Tensor>> x) = 0; // pure virtual function

                virtual Tensor get_detached_value(Tensor& t) = 0;

                virtual void initialize_parameters() {}


                // TODO: correct this as the implementation was copied from the Matrix class that we do not use right now
                static void xavier_init(Tensor& t, int input_size, int output_size) {

                    std::random_device rd{};
                    std::mt19937 gen{rd()};


                    float stddev = sqrt(2.0 / (input_size + output_size));

                    std::normal_distribution<double> d{0.0,stddev};

                    // Case 1: vector or scalar

                    // Case 2: matrix
                    ll batches = t.sizeOfTensor();
                    int rows = t.getShape()[t.getShape().size()-2];
                    int cols = t.getShape()[t.getShape().size()-1];
                    batches/= ( rows*cols );

                    for (ll b = 0; b <batches; b++){
                        for (int i = 0; i < rows; i++) {
                            for (int j = 0; j < cols; j++) {
                                t.set_with_offset(b, i,j, d(gen));
                            }
                        }
                    }
                }

                // He initialization -> TODO: will implement
                static void he_init(Tensor& tensor, int input_size);

            };


        // inherits operations from Mod and also gets structure from Module class
        class Linear : Module{
            private:
                std::shared_ptr<simplenet::Node<Tensor>> W;  // Weight matrix as a node for autogradient
                std::shared_ptr<simplenet::Node<Tensor>> B;  // Bias vector as a node for autogradient
                int input_size;
                int output_size;
                std::string initialization_method;
            public:
                Linear(int in_shape, int out_shape, std::string initialization = "Xavier") : input_size(in_shape), output_size(out_shape), initialization_method(initialization){
                    Tensor weight_tensor({input_size, output_size});
                    W = simplenet::Node<Tensor>::make_node(weight_tensor);

                    Tensor bias_tensor({ output_size});
                    B = simplenet::Node<Tensor>::make_node(bias_tensor);
                    initialize_parameters();
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<Tensor>> forward(std::shared_ptr<simplenet::Node<Tensor>> x) override{
                    return x * W + B; // convert input shape to output shape
                }

                Tensor get_detached_value(Tensor& t)override {
                    return (t*W +B)->val;
                }

                // we will perform Xavier Init here
                void initialize_parameters() override{
                    // default initialization is always 0 as we already know as that is what our Tensor class does
                    if (this->initializaion_method == "Zero") {}
                    else if (this->initialization_method == "Xavier") this->xavier_init(W->val, input_size, output_size);
                    // we dont initialize the bias Tensor at the moment
                };

                // Helpers
                std::shared_ptr<simplenet::Node<Tensor>> get_weights() const { return W; }
                std::shared_ptr<simplenet::Node<Tensor>> get_bias() const { return B; }

                int get_in_shape() const { return input_size; }
                int get_out_shape() const { return output_size; }
        };
    }

    namespace activations_function {


    }
}
#endif
