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

                    *p -= learning_rate *( m_hat / (T::sqrt(v_hat) + eps));
                }
                this->step_count++;
            }

            template<typename T>
            void Adam<T>::zero_grad(){
                for (auto& p : this->params){
                    p->zero_grad();
                }
            }

            // ---- AdamW definitions ----
            template<typename T>
            AdamW<T>::AdamW(std::vector<std::shared_ptr<T>> params, double learning_rate, double beta1, double beta2, double eps, double weight_decay)
                : params(params), learning_rate(learning_rate), beta1(beta1), beta2(beta2), eps(eps), step_count(1), weight_decay(weight_decay){
                for (auto& p : params) {
                    m.push_back(T(p->getShape(), p->getDevice())); // zeros, same shape and device as param
                    v.push_back(T(p->getShape(), p->getDevice()));
                }
            }

            template<typename T>
            void AdamW<T>::step(){
                for (size_t i = 0; i < params.size(); i++) {
                    auto& p = params[i];
                    m[i] = beta1 * m[i] + (1 - beta1) * p->grad(); // scalar multiplication
                    v[i] = beta2 * v[i] + (1 - beta2) * bearml::linear_algebra::hadamard(p->grad(), p->grad()); // element-wise square

                    // Bias-corrected estimates
                    T m_hat = m[i] / (1 - std::pow(beta1, step_count));
                    T v_hat = v[i] / (1 - std::pow(beta2, step_count));

                    *p -= learning_rate *( m_hat / (T::sqrt(v_hat) + eps) + weight_decay* (*p));
                }
                this->step_count++;
            }

            template<typename T>
            void AdamW<T>::zero_grad(){
                for (auto& p : this->params){
                    p->zero_grad();
                }
            }


            // Muon optimizer
            // source: https://docs.pytorch.org/docs/2.15/generated/torch.optim.Muon.html
            template<typename T>
            Muon<T>::Muon(std::vector<std::shared_ptr<T>> params, double lr, double wd, double momemtum, bool nesterov, std::vector<double> ns_coefficients, double eps, int64_t ns_steps):
            params(params), learning_rate(lr), weight_decay(wd), momentum(momemtum), nesterov(nesterov), ns_coefficients(ns_coefficients),eps(eps), ns_steps(ns_steps)  {
                // initialize momentum buffer to 0
                for (auto& p : params) {
                    momentum_buffer.push_back(T(p->getShape(), p->getDevice())); // zeros, same shape and device as param
                }
            }

            // template<typename T>
            // T Muon<T>::NewtonSchulz(std::shared_ptr<T> x) {
            //     // formula is X_k+1 = 1/2 X_k (3I-X_k A X_k)
            //     for (int i =0; i<ns_steps; i++){
            //         x = 0.5*x*(3 T::iden)
            //     }
            // }


            template<typename T>
            void Muon<T>::step(){
                for (size_t i = 1; i < params.size(); i++) {
                    momentum_buffer[i] = momentum*momentum_buffer[i-1] + params[i]->grad;
                    if (nesterov) {
                        momentum_buffer[i] =  momentum*momentum_buffer[i] + params[i]->grad;
                    }

                    // Perform Newton-Schulz
                }
            }



        }
    }
}

template class bearml::neural_network::optimizers::SGD<bearml::Tensorf>;
template class bearml::neural_network::optimizers::SGD<bearml::TensorD>;
template class bearml::neural_network::optimizers::Adam<bearml::Tensorf>;
template class bearml::neural_network::optimizers::Adam<bearml::TensorD>;
template class bearml::neural_network::optimizers::AdamW<bearml::Tensorf>;
template class bearml::neural_network::optimizers::AdamW<bearml::TensorD>;
