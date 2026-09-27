#pragma once
#include "utils/dtype.h"
#include <type_traits>

namespace bearml {
    template<typename T>
    class Scalar {
        private:
            T scalar{};
        public:
            Scalar() = default;
            Scalar(T scalar): scalar(scalar){
            }

            // other arithmetic types (e.g. a double literal into a bfloat16 scalar) - explicit cast, no narrowing warning
            template<typename U> requires (std::is_arithmetic_v<U> && !std::is_same_v<U, T>)
            Scalar(U other): scalar(static_cast<T>(other)){
            }

            operator T() const { return scalar; }
            T value() const { return scalar; }

            Scalar(const Scalar<T>& other) = default;
            Scalar& operator=(const Scalar<T>& other) = default;
            Scalar(Scalar<T>&& other) = default;
            Scalar& operator=(Scalar<T>&& other) = default;
            ~Scalar() = default;

            // inherit ops
    };
}
