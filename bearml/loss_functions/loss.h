#pragma once

#include "operators/ops.h"
#include "tensor/Tensor.h"
#include "autograd/autogradient.h"
#include <stdexcept>

#ifndef LOSS_H
#define LOSS_H

namespace bearml {
    namespace neural_network {
        namespace loss_functions {
            // --------------------------------------- Convex loss-functions ----------------------------------------
            // Mean Absolute error - works
            template<typename T>
            std::shared_ptr<bearml::Node<T>> l1_loss(std::shared_ptr<bearml::Node<T>> actual, std::shared_ptr<bearml::Node<T>> predictions){

                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD)");

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
            std::shared_ptr<bearml::Node<T>> l2_loss( std::shared_ptr<bearml::Node<T>> actual,  std::shared_ptr<bearml::Node<T>> predictions){
                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD)");

                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto sqr_diff = bearml::Node<T>::hadamard(diff,diff); // should be implemented now
                auto loss = mean(sqr_diff);         // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // Log loss
            template<typename T>
            std::shared_ptr<bearml::Node<T>> log_loss(std::shared_ptr<bearml::Node<T>> actual, std::shared_ptr<bearml::Node<T>> predictions){

                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD / TensorBF)");

                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto log_p = bearml::Node<T>::log(predictions);
                auto log_1_minus_p = bearml::Node<T>::log(1.0 - predictions);

                // This is the hadamard product of actual and log(predictions), and (1-actual) and log(1-predictions)
                auto term1 = bearml::Node<T>::hadamard(actual, log_p);
                auto term2 = bearml::Node<T>::hadamard(1.0 - actual, log_1_minus_p);

                auto logloss = -1.0 * (term1 + term2);

                auto loss = mean(logloss);

                return loss;
            }

            // BCE
            template<typename T>
            std::shared_ptr<bearml::Node<T>> bce_loss_with_logits(std::shared_ptr<bearml::Node<T>> actual, std::shared_ptr<bearml::Node<T>> predictions){
                return log_loss<T>(actual, predictions);
            }


            template <typename T>
            std::shared_ptr<bearml::Node<T>> cross_entropy_loss(std::shared_ptr<bearml::Node<T>> actual, std::shared_ptr<bearml::Node<T>> predictions){
                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD / TensorBF)");

                if (actual->val.getShape() != predictions->val.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                int dim = static_cast<int>(actual->val.getShape().size()) - 1; // last dim is class dim

                auto p = bearml::Node<T>::softmax(predictions, dim);
                auto log_p = bearml::Node<T>::log(p);
                auto ce = -1.0 * bearml::Node<T>::hadamard(actual, log_p);
                auto per_sample_loss = bearml::Node<T>::sum(ce, dim); // sum over class dim (keepdims=true)
                auto loss = mean(per_sample_loss);                    // mean over remaining (batch) dims
                return loss;

            }

        }
    }
}
#endif
