#pragma once

#include "tensor/Tensor.h"
#include "autograd/autogradient.h"
#include <stdexcept>

#ifndef LOSS_H
#define LOSS_H

namespace simplenet {
    namespace neural_network {
        namespace loss_functions {
            // --------------------------------------- Convex loss-functions ----------------------------------------
            // Mean Absolute error - works
            std::shared_ptr<simplenet::Node<simplenet::Tensor>> l1_loss(std::shared_ptr<simplenet::Node<simplenet::Tensor>> actual, std::shared_ptr<simplenet::Node<simplenet::Tensor>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto abs_diff = abs(diff);          // element-wise absolute value
                auto loss = mean(abs_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // Mean Squared error - TODO: check this if it works now
            std::shared_ptr<simplenet::Node<simplenet::Tensor>> l2_loss( std::shared_ptr<simplenet::Node<simplenet::Tensor>> actual,  std::shared_ptr<simplenet::Node<simplenet::Tensor>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto sqr_diff = simplenet::Node<simplenet::Tensor>::hadamard(diff,diff); // should be implemented now
                auto loss = mean(sqr_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }




            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // Log loss
            std::shared_ptr<simplenet::Node<simplenet::Tensor>> log_loss(std::shared_ptr<simplenet::Node<simplenet::Tensor>> actual, std::shared_ptr<simplenet::Node<simplenet::Tensor>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto logloss = -1.0 * (actual * simplenet::Node<simplenet::Tensor>::log(predictions)) + (1.0 - actual) * simplenet::Node<simplenet::Tensor>::log(1.0 - predictions);
                auto loss = mean(logloss);

                return loss;
            }
            // BCE
            inline std::shared_ptr<simplenet::Node<simplenet::Tensor>> bce_loss(std::shared_ptr<simplenet::Node<simplenet::Tensor>> actual, std::shared_ptr<simplenet::Node<simplenet::Tensor>> predictions, std::string){
                return log_loss(actual, predictions);
            }


            // Cross Entropy
            void cross_entropy_loss(int actual, int predictions){
                // Log loss
            }

        }
    }
}
#endif
