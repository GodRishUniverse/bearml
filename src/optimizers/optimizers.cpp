#include "optimizers.h"

namespace simplenet {
    namespace neural_network{
        namespace optimizers {
            // TODO: Complete these optimizers
           SGD::SGD(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate) : params(params), learning_rate(learning_rate){

           }

           void SGD::step(){
               // TODO
           }


           Adam::Adam(std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> params, double learning_rate, double momentum) : params(params), learning_rate(learning_rate), momentum(momentum){

           }

           void Adam::step(){
               // TODO
           }
        }
    }
}
