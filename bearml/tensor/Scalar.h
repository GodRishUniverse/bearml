#pragma once

namespace bearml {

    template<typename T>
    class Scalar {
        private:
            T scalar;
        public:
            Scalar() = default;
            Scalar(T v) : scalar(v) {}
            T scalar() const { return scalar; }
            operator T() const { return scalar; }

            // inherit ops

    };
}
