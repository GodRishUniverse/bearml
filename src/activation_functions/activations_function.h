#pragma once
#include <vector>
#include <stdexcept>
#include <random>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"

#ifndef ACTIVATION_FUNCTIONS
namespace simplenet {
    class Tensor; // forward declaration
    namespace activations_function {
        // will apply polymorphism and inheritance for activation functions as the principle idea is the same
        // will inherit the Node class for operations to be carried out and use the backward features

        // No name namespace for module abstract class
        namespace {
            class Module{
                public:
                    virtual ~Module() {}; // it is a pure virtual class
                    virtual simplenet::autogradient::Node<Tensor> forward() = 0; // pure virtual function

                    // TODO: correct this as the implementation was copied from the Matrix class that we do not use right now
                    static void xavier_init(Tensor& t, int inrows, int incols, int input_size, int output_size) {

                        std::random_device rd{};
                        std::mt19937 gen{rd()};


                        float stddev = sqrt(2.0 / (input_size + output_size));

                        std::normal_distribution<double> d{0.0,stddev};

                        Matrix<T> m(inrows, incols);
                        for (int i = 0; i < inrows; i++) {
                            for (int j = 0; j < incols; j++) {
                                m.set(i, j, static_cast<T>(d(gen)));
                            }
                        }
                        return m;
                    }
            };
        }

        // inherits operations from Node and also gets structure from Module class
        class Linear : Module, simplenet::autogradient::Node<Tensor>{
            private:
                Tensor W;
                Tensor B;
            public:
                Linear(std::vector<int>& in_shape, std::vector<int>& out_shape): W(in_shape), B(out_shape){
                    // we will xavier initialize these matrices
                    this->xavier_init(W, int inrows, int incols, int input_size, int output_size);
                    this->xavier_init(B, int inrows, int incols, int input_size, int output_size);
                }

                // need to check if this is the correct way to get the forward declaration
                simplenet::autogradient::Node<Tensor> forward() {

                }


        };
    }
}
#endif
