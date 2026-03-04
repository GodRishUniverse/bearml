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
           Adam::Adam(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate, double momentum, double rms_prop, double beta1,  double beta2, double eps) : params(params), learning_rate(learning_rate), momentum(momentum), rms_prop(rms_prop), beta1(beta1), beta2(beta2), eps(eps), step_count(1){

           }

           void Adam::step(){
               for (auto p : this->params){
                   momentum = beta1*momentum+(1-beta1)*p->grad;
                   rms_prop = beta2*rms_prop+(1-beta2)*p->grad*p->grad;

                   // bias correction
                   double m_hat = momentum/(1-pow(beta1, this->step_count));
                   double v_hat = rms_prop/(1-pow(beta2, this->step_count));
                   p->val -= learning_rate*m_hat/(std::sqrt(v_hat)+eps);
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
