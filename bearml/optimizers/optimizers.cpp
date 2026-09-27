// SGD and Adam definitions - declared in optimizers.h, instantiated at the bottom
#include "optimizers/optimizers.h"

namespace bearml {
    namespace neural_network{
        namespace optimizers {
            // ---- SGD definitions ----
            template<typename T>
            SGD<T>::SGD(std::vector<std::shared_ptr<T>> params, double learning_rate)
                : params(params), learning_rate(learning_rate) {}

            // Use this after doing backward pass on the computational graph
            template<typename T>
            void SGD<T>::step(){
                for (auto& p : this->params){   // shared pointers - a copy of a tensor is deep and untracked
                    *p -= learning_rate*p->grad();
                }
            }

            template<typename T>
            void SGD<T>::zero_grad(){
                for (auto& p : this->params){
                    p->zero_grad();
                }
            }

            // ---- Adam definitions ----
            // src - https://builtin.com/machine-learning/adam-optimization
            template<typename T>
            Adam<T>::Adam(std::vector<std::shared_ptr<T>> params, double learning_rate, double beta1, double beta2, double eps)
                : params(params), learning_rate(learning_rate), beta1(beta1), beta2(beta2), eps(eps), step_count(1){
                for (auto& p : params) {
                    m.push_back(T(p->getShape(), p->getDevice())); // zeros, same shape and device as param
                    v.push_back(T(p->getShape(), p->getDevice()));
                }
            }

            template<typename T>
            void Adam<T>::step(){
                for (size_t i = 0; i < params.size(); i++) {
                    auto& p = params[i];
                    m[i] = beta1 * m[i] + (1 - beta1) * p->grad(); // scalar multiplication
                    v[i] = beta2 * v[i] + (1 - beta2) * bearml::linear_algebra::hadamard(p->grad(), p->grad()); // element-wise square

                    // Bias-corrected estimates
                    T m_hat = m[i] / (1 - std::pow(beta1, step_count));
                    T v_hat = v[i] / (1 - std::pow(beta2, step_count));

                    *p -= learning_rate * m_hat / (T::sqrt(v_hat) + eps);
                }
                this->step_count++;
            }

            template<typename T>
            void Adam<T>::zero_grad(){
                for (auto& p : this->params){
                    p->zero_grad();
                }
            }
        }
    }
}

template class bearml::neural_network::optimizers::SGD<bearml::Tensorf>;
template class bearml::neural_network::optimizers::SGD<bearml::TensorD>;
template class bearml::neural_network::optimizers::Adam<bearml::Tensorf>;
template class bearml::neural_network::optimizers::Adam<bearml::TensorD>;
