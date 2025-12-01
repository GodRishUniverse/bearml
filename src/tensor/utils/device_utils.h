#pragma once
#include <stdexcept>

namespace simplenet {
    class Tensor; // forward declaration
    namespace utils {
        void errorCheckSameDevice(const Tensor& a, const Tensor& b) {
            if (a.getDevice() != b.getDevice()) {
                throw std::invalid_argument(
                    "Tensors must be on same device. "
                );
            }
        }

    }
}
