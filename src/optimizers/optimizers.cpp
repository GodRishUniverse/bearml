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

           // TODO: Complete these optimizers
           Adam::Adam(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate, double momentum) : params(params), learning_rate(learning_rate), momentum(momentum){

           }

           void Adam::step(){
               // TODO
           }

           void Adam::zero_grad(){
                for (auto p : this->params){
                    p->grad.fill(0.0);
                }
           }
        }
    }
}
