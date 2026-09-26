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
            T l1_loss(T& actual, T& predictions){

                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD)");

                if (actual.getShape() != predictions.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto abs_diff = T::abs(diff);       // element-wise absolute value
                auto loss = T::mean(abs_diff);      // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // Mean Squared error - TODO: check this if it works now
            template<typename T>
            T l2_loss( T& actual,  T& predictions){
                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD)");

                if (actual.getShape() != predictions.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto diff = predictions - actual;   // element-wise difference
                auto sqr_diff = bearml::linear_algebra::hadamard(diff,diff); // should be implemented now
                auto loss = T::mean(sqr_diff);      // mean of all absolute differences

                return loss; // now can be done backward here
            }

            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // Log loss
            template<typename T>
            T log_loss(T& actual, T& predictions){

                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD / TensorBF)");

                if (actual.getShape() != predictions.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                auto log_p = T::log(predictions);
                auto one_minus_p = 1.0 - predictions;   // named so T::log can take it by reference
                auto log_1_minus_p = T::log(one_minus_p);

                // This is the hadamard product of actual and log(predictions), and (1-actual) and log(1-predictions)
                auto term1 = bearml::linear_algebra::hadamard(actual, log_p);
                auto one_minus_actual = 1.0 - actual;
                auto term2 = bearml::linear_algebra::hadamard(one_minus_actual, log_1_minus_p);

                auto logloss = -1.0 * (term1 + term2);

                auto loss = T::mean(logloss);

                return loss;
            }

            // BCE
            template<typename T>
            T bce_loss_with_logits(T& actual, T& predictions){
                return log_loss<T>(actual, predictions);
            }


            template <typename T>
            T cross_entropy_loss(T& actual, T& predictions){
                static_assert(bearml::is_tensor_v<T>,
                    "loss functions require a Tensor type (e.g. Tensorf / TensorD / TensorBF)");

                if (actual.getShape() != predictions.getShape()){
                    throw std::runtime_error("Shapes of actual and predictions do not match!");
                }

                int dim = static_cast<int>(actual.getShape().size()) - 1; // last dim is class dim

                auto p = T::softmax(predictions, dim);
                auto log_p = T::log(p);
                auto ce = -1.0 * bearml::linear_algebra::hadamard(actual, log_p);
                auto per_sample_loss = ce.accumulate(dim, bearml::reductions::ReductionOps::SUM, true); // sum over class dim (keepdims=true)
                auto loss = T::mean(per_sample_loss);                    // mean over remaining (batch) dims
                return loss;

            }

        }
    }
}
#endif
