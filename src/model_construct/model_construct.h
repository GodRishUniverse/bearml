#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"

using ll = long long;

#ifndef MODEL_CONSTRUCT
#define MODEL_CONSTRUCT


// idea here is to make a class that acts like the nn.Module in PyTorch

namespace simplenet{
    namespace neural_network {
        class Model_Construct{
            // TODO: might update this to get the grads and weights for each node (except the inputs) for the updation of weight matrices
            public:
                virtual ~Model_Construct()  = default; // it is a pure virtual class

                // we will be passing by value here as we are not sure what the user passes in
                virtual std::shared_ptr<simplenet::Node<simplenet::Tensor>> forward(std::vector<simplenet::Tensor> inputs) = 0; // pure virtual function

                // get the parameters of the model class
                // pointers ensure that when they are updated the same memory block is updated and there are no duplicates
                virtual std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> parameters() = 0;

        };
    }
}

#endif
