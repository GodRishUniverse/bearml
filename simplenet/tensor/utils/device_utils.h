#pragma once
#include <stdexcept>

namespace simplenet {
    template<typename T> class Tensor; // forward declaration
    namespace utils {
        // templated: works for any Tensor<T> element type. Defined inline because
        // it is a template (instantiated lazily once Tensor<T> is a complete type).
        template<typename T>
        inline void errorCheckSameDevice(const Tensor<T>& a, const Tensor<T>& b) {
            if (a.getDevice() != b.getDevice()) {
                throw std::invalid_argument(
                    "Tensors must be on same device. "
                );
            }
        }

    }
}
