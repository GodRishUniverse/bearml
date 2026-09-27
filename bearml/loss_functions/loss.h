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
            T l1_loss(T& actual, T& predictions);

            // Mean Squared error - TODO: check this if it works now
            template<typename T>
            T l2_loss( T& actual,  T& predictions);

            // --------------------------------------- Non-Convex loss-functions ----------------------------------------

            // Log loss
            template<typename T>
            T log_loss(T& actual, T& predictions);

            // BCE
            template<typename T>
            T bce_loss_with_logits(T& actual, T& predictions);


            template <typename T>
            T cross_entropy_loss(T& actual, T& predictions);

        }
    }
}
#endif
