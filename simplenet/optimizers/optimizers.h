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
            template<typename T>
            class SGD: public Optimizer{
                    static_assert(simplenet::is_tensor_v<T>,
                        "SGD requires a Tensor type (e.g. Tensorf / TensorD)");
                    static_assert(simplenet::is_supported_float_v<simplenet::tensor_element_t<T>>,
                        "SGD parameters must be float, double, or bfloat16 tensors");
                private:
                    std::vector<std::shared_ptr<simplenet::Node<T>>> params;
                    double learning_rate;
                public:
                    SGD(std::vector<std::shared_ptr<simplenet::Node<T>>> params, double learning_rate= 0.0001);
                    void step() override;
                    void zero_grad() override;
            };


            // ADAM optimizer
            template <typename T>
            class Adam: public Optimizer{
                    static_assert(simplenet::is_tensor_v<T>,
                        "Adam requires a Tensor type (e.g. Tensorf / TensorD)");
                    static_assert(simplenet::is_supported_float_v<simplenet::tensor_element_t<T>>,
                        "Adam parameters must be float, double, or bfloat16 tensors");
                private:
                    std::vector<std::shared_ptr<simplenet::Node<T>>> params;
                    std::vector<T> m; // momentum
                    double learning_rate;
                    std::vector<T> v; // rms_prop
                    double beta1; // decay rate for momentum
                    double beta2; // decay rate for RMSProp
                    double eps; // small value to avoid division by zero

                    int64_t step_count;

                public:
                    Adam(std::vector<std::shared_ptr<simplenet::Node<T>>> params, double learning_rate= 0.0001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
                    void step() override;
                    void zero_grad() override;
            };

            // ---- SGD definitions (header-only: template members must be visible at instantiation) ----
            template<typename T>
            SGD<T>::SGD(std::vector<std::shared_ptr<simplenet::Node<T>>> params, double learning_rate)
                : params(params), learning_rate(learning_rate) {}

            // Use this after doing backward pass on the computational graph
            template<typename T>
            void SGD<T>::step(){
                for (auto p : this->params){
                    p->val -= learning_rate*p->grad;
                }
            }

            template<typename T>
            void SGD<T>::zero_grad(){
                for (auto p : this->params){
                    p->grad.fill(0.0);
                }
            }

            // ---- Adam definitions ----
            // src - https://builtin.com/machine-learning/adam-optimization
            template<typename T>
            Adam<T>::Adam(std::vector<std::shared_ptr<simplenet::Node<T>>> params, double learning_rate, double beta1, double beta2, double eps)
                : params(params), learning_rate(learning_rate), beta1(beta1), beta2(beta2), eps(eps), step_count(1){
                for (auto& p : params) {
                    m.push_back(T(p->val.getShape())); // zeros, same shape as param
                    v.push_back(T(p->val.getShape()));
                }
            }

            template<typename T>
            void Adam<T>::step(){
                for (size_t i = 0; i < params.size(); i++) {
                    auto& p = params[i];
                    m[i] = beta1 * m[i] + (1 - beta1) * p->grad; // scalar multiplication
                    v[i] = beta2 * v[i] + (1 - beta2) * simplenet::linear_algebra::hadamard(p->grad, p->grad); // element-wise square

                    // Bias-corrected estimates
                    T m_hat = m[i] / (1 - std::pow(beta1, step_count));
                    T v_hat = v[i] / (1 - std::pow(beta2, step_count));

                    p->val -= learning_rate * m_hat / (T::sqrt(v_hat) + eps);
                }
                this->step_count++;
            }

            template<typename T>
            void Adam<T>::zero_grad(){
                for (auto p : this->params){
                    p->grad.fill(0.0);
                }
            }
        }
    }
}


#endif
