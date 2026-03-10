#include "optimizers.h"

namespace simplenet {
    namespace neural_network{
        namespace optimizers {
           // The Stochastic part comes from the random selection of the batches
           SGD::SGD(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate) : params(params), learning_rate(learning_rate){

           }

           // Use this after doing backward pass on the computational graph
           void SGD::step(){
               for (auto p : this->params){
                   p->val -= learning_rate*p->grad;
               }
           }

           void SGD::zero_grad(){
               for (auto p : this->params){
                   p->grad.fill(0.0);
               }
           }

           // TODO: fix Adam implementation
           // src - https://builtin.com/machine-learning/adam-optimization#:~:text=Momentum%20speeds%20up%20training%20by,need%20to%20take%20fewer%20steps.
           Adam::Adam(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate, double beta1,  double beta2, double eps) : params(params), learning_rate(learning_rate), beta1(beta1), beta2(beta2), eps(eps), step_count(1){
               for (auto& p : params) {
                      m.push_back(simplenet::Tensor(p->val.getShape())); // zeros, same shape as param
                      v.push_back(simplenet::Tensor(p->val.getShape()));
                }
           }

           void Adam::step(){
               for (size_t i = 0; i < params.size(); i++) {
                   auto& p = params[i];
                   m[i] = beta1 * m[i] + (1 - beta1) * p->grad; // hadamard here cause scalar multiplication
                   v[i] = beta2 * v[i] + (1 - beta2) * simplenet::linear_algebra::hadamard(p->grad, p->grad); // hadamard - explicit hadamard product needed for element-wise multiplication

                   // Bias-corrected estimates
                   Tensor m_hat = m[i] / (1 - std::pow(beta1, step_count)); // element wise division
                   Tensor v_hat = v[i] / (1 - std::pow(beta2, step_count)); // element wise division

                   p->val -= learning_rate * m_hat / (simplenet::Tensor::sqrt(v_hat) + eps); // TODO: write a function for sqrt for Tensor - element wise
               }
               this->step_count++;
           }

           void Adam::zero_grad(){
                for (auto p : this->params){
                    p->grad.fill(0.0);
                }
           }
        }
    }
}
