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

            // Mean Squared error
            // std::shared_ptr<simplenet::Node<simplenet::Tensor>> l2_loss( std::shared_ptr<simplenet::Node<simplenet::Tensor>> actual,  std::shared_ptr<simplenet::Node<simplenet::Tensor>> predictions){
            //     // if (actual->val.getShape() != predictions->val.getShape()){
            //     //     throw std::runtime_error("Shapes of actual and predictions do not match!");
            //     // }

            //     // // TODO: Implement backward op for hadamard multiplication
            //     // auto diff = predictions - actual;   // element-wise difference
            //     // auto sqr_diff = simplenet::linear_algebra::hadamard(diff,diff); // TODO: <- implement backward op
            //     // auto loss = mean(sqr_diff);         // mean of all absolute differences

            //     // return loss; // now can be done backward here
            // }

            void log_loss(int actual, int predictions){
                // Log loss
            }

            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // BCE
            void bce_loss(int actual, int predictions){
                // Log loss
            }


            // Cross Entropy
            void cross_entropy_loss(int actual, int predictions){
                // Log loss
            }

        }
    }
}
#endif
