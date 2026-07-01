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
            template<typename T>
            std::shared_ptr<simplenet::Node<T>> l1_loss(std::shared_ptr<simplenet::Node<T>> actual, std::shared_ptr<simplenet::Node<T>> predictions){

                if (!simplenet::is_tensor_v<T>) {
                    throw std::runtime_error("T must be a tensor type!");
                }

                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto abs_diff = abs(diff);          // element-wise absolute value
                auto loss = mean(abs_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // Mean Squared error - TODO: check this if it works now
            template<typename T>
            std::shared_ptr<simplenet::Node<T>> l2_loss( std::shared_ptr<simplenet::Node<T>> actual,  std::shared_ptr<simplenet::Node<T>> predictions){
                if (!simplenet::is_tensor_v<T>) {
                    throw std::runtime_error("T must be a tensor type!");
                }

                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto sqr_diff = simplenet::Node<T>::hadamard(diff,diff); // should be implemented now
                auto loss = mean(sqr_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // Log loss
            template<typename T>
            std::shared_ptr<simplenet::Node<T>> log_loss(std::shared_ptr<simplenet::Node<T>> actual, std::shared_ptr<simplenet::Node<T>> predictions){

                if (!simplenet::is_tensor_v<T>) {
                    throw std::runtime_error("T must be a tensor type!");
                }

                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto log_p = simplenet::Node<T>::log(predictions);
                auto log_1_minus_p = simplenet::Node<T>::log(1.0 - predictions);

                // This is the hadamard product of actual and log(predictions), and (1-actual) and log(1-predictions)
                auto term1 = simplenet::Node<T>::hadamard(actual, log_p);
                auto term2 = simplenet::Node<T>::hadamard(1.0 - actual, log_1_minus_p);

                auto logloss = -1.0 * (term1 + term2);

                auto loss = mean(logloss);

                return loss;
            }

            // BCE
            template<typename T>
            std::shared_ptr<simplenet::Node<T>> bce_loss_with_logits(std::shared_ptr<simplenet::Node<T>> actual, std::shared_ptr<simplenet::Node<T>> predictions, std::string){
                return log_loss<T>(actual, predictions);
            }


            // Cross Entropy
            void cross_entropy_loss(int actual, int predictions){
                // Log loss
            }

        }
    }
}
#endif
