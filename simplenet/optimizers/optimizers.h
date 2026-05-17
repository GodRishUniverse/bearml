#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

namespace simplenet {
    namespace neural_network{
        namespace optimizers {
            class Optimizer {
                public:
                    virtual ~Optimizer() = default;
                    virtual void step() = 0; // pure virtual function
                    virtual void zero_grad() = 0; // pure virtual function
            };

            // Class definitions with default values to not get the errors -> need to add regularization and eps, betas
            class SGD: public Optimizer{
                private:
                    std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> params;
                    double learning_rate;
                public:
                    SGD(std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> params, double learning_rate= 0.0001);
                    void step() override;
                    void zero_grad() override;
            };


            // ADAM optimizer
            class Adam: public Optimizer{
                private:
                    std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> params;
                    std::vector<simplenet::TensorD> m; // momentum
                    double learning_rate;
                    std::vector<simplenet::TensorD> v; // rms_prop
                    double beta1; // decay rate for momentum
                    double beta2; // decay rate for RMSProp
                    double eps; // small value to avoid division by zero

                    int64_t step_count;

                public:
                    Adam(std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> params, double learning_rate= 0.0001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
                    void step() override;
                    void zero_grad() override;
            };
        }
    }
}


#endif
