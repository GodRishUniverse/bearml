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
            std::shared_ptr<simplenet::Node<simplenet::TensorD>> l1_loss(std::shared_ptr<simplenet::Node<simplenet::TensorD>> actual, std::shared_ptr<simplenet::Node<simplenet::TensorD>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto abs_diff = abs(diff);          // element-wise absolute value
                auto loss = mean(abs_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // Mean Squared error - TODO: check this if it works now
            std::shared_ptr<simplenet::Node<simplenet::TensorD>> l2_loss( std::shared_ptr<simplenet::Node<simplenet::TensorD>> actual,  std::shared_ptr<simplenet::Node<simplenet::TensorD>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto sqr_diff = simplenet::Node<simplenet::TensorD>::hadamard(diff,diff); // should be implemented now
                auto loss = mean(sqr_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // Log loss
            std::shared_ptr<simplenet::Node<simplenet::TensorD>> log_loss(std::shared_ptr<simplenet::Node<simplenet::TensorD>> actual, std::shared_ptr<simplenet::Node<simplenet::TensorD>> predictions){
                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto log_p = simplenet::Node<simplenet::TensorD>::log(predictions);
                auto log_1_minus_p = simplenet::Node<simplenet::TensorD>::log(1.0 - predictions);

                // This is the hadamard product of actual and log(predictions), and (1-actual) and log(1-predictions)
                auto term1 = simplenet::Node<simplenet::TensorD>::hadamard(actual, log_p);
                auto term2 = simplenet::Node<simplenet::TensorD>::hadamard(1.0 - actual, log_1_minus_p);

                auto logloss = -1.0 * (term1 + term2);

                auto loss = mean(logloss);

                return loss;
            }

            // BCE
            inline std::shared_ptr<simplenet::Node<simplenet::TensorD>> bce_loss_with_logits(std::shared_ptr<simplenet::Node<simplenet::TensorD>> actual, std::shared_ptr<simplenet::Node<simplenet::TensorD>> predictions, std::string){
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
