#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"

#ifndef SGD_H
#define SGD_H

namespace simplenet {
    namespace neural_network{
        namespace optimizers {
            class Optimizer {
                public:
                    virtual ~Optimizer() = default;
                    virtual void step() = 0; // pure virtual function
            };

            // Class definitions with default values to not get the errors -> need to add regularization and eps, betas
            class SGD: public Optimizer{
                private:
                    std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params;
                    double learning_rate;
                public:
                    SGD(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate= 0.0001);
                    void step() override;
            };

            class Adam: public Optimizer{
                private:
                    std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params;
                    double momentum;
                    double learning_rate;

                public:
                    Adam(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate= 0.0001, double momentum = 0.001);
                    void step() override;
            };
        }
    }
}


#endif
