#pragma once
#include <vector>
#include <stdexcept>

#ifndef ACTIVATION_FUNCTIONS
namespace simplenet {
    class Tensor; // forward declaration
    namespace activations_function {
        // will apply polymorphism and inheritance for activation functions as the principle idea is the same
        // will inherit the Node class for operations to be carried out and use the backward features
    }
}
#endif
