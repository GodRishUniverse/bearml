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

namespace bearml {
    namespace neural_network{
        namespace optimizers {
            class Optimizer {
                public:
                    virtual ~Optimizer() = default;
                    virtual void step() = 0; // pure virtual function
                    virtual void zero_grad() = 0; // pure virtual function
            };

            // Class definitions with default values to not get the errors -> need to add regularization and eps, betas
            template<typename T>
            class SGD: public Optimizer{
                    static_assert(bearml::is_tensor_v<T>,
                        "SGD requires a Tensor type (e.g. Tensorf / TensorD)");
                    static_assert(bearml::is_floating(bearml::dtype_of<bearml::tensor_element_t<T>>),
                        "SGD parameters must be float, double, or bfloat16 tensors");
                private:
                    std::vector<std::shared_ptr<T>> params;
                    double learning_rate;
                public:
                    SGD(std::vector<std::shared_ptr<T>> params, double learning_rate= 0.0001);
                    void step() override;
                    void zero_grad() override;
            };


            // ADAM optimizer
            template <typename T>
            class Adam: public Optimizer{
                    static_assert(bearml::is_tensor_v<T>,
                        "Adam requires a Tensor type (e.g. Tensorf / TensorD)");
                    static_assert(bearml::is_floating(bearml::dtype_of<bearml::tensor_element_t<T>>),
                        "Adam parameters must be float, double, or bfloat16 tensors");
                private:
                    std::vector<std::shared_ptr<T>> params;
                    std::vector<T> m; // momentum
                    double learning_rate;
                    std::vector<T> v; // rms_prop
                    double beta1; // decay rate for momentum
                    double beta2; // decay rate for RMSProp
                    double eps; // small value to avoid division by zero

                    int64_t step_count;

                public:
                    Adam(std::vector<std::shared_ptr<T>> params, double learning_rate= 0.0001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
                    void step() override;
                    void zero_grad() override;
            };

            // bodies live in optimizers.cpp, instantiated there for Tensorf and TensorD
            extern template class SGD<bearml::Tensorf>;
            extern template class SGD<bearml::TensorD>;
            extern template class Adam<bearml::Tensorf>;
            extern template class Adam<bearml::TensorD>;
        }
    }
}


#endif
