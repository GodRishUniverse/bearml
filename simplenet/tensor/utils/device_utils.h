#pragma once
#include <stdexcept>

namespace simplenet {
    class Tensor; // forward declaration
    namespace utils {
        void errorCheckSameDevice(const Tensor& a, const Tensor& b) ;

    }
}
