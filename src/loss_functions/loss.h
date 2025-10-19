#pragma once

#include "tensor/Tensor.h"
#include "autograd/autogradient.h"
#include <stdexcept>

#ifndef LOSS_H
#define LOSS_H

namespace simplenet {
    namespace neural_network {
        namespace loss_functions {
            // IMPLEMENT LOSS FUNCTIONS

            // Mean Absolute error
            std::shared_ptr<simplenet::Node<simplenet::Tensor>> l1_loss(std::shared_ptr<simplenet::Node<simplenet::Tensor>> actual, std::shared_ptr<simplenet::Node<simplenet::Tensor>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                // TODO: Implement abs and mean in Tensor and then Node
                auto diff = predictions - actual;   // element-wise difference
                auto abs_diff = abs(diff);          // element-wise absolute value
                auto loss = mean(abs_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }


            void l2_loss(int actual, int predictions){
                // MSE
            }

            void log_loss(int actual, int predictions){
                // Log loss
            }


        }
    }
}
#endif
