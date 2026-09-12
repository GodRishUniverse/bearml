#pragma once

namespace bearml {
    template<typename T>
    class Scalar {
        private:
            T scalar{};
        public:
            Scalar() = default;
            Scalar(T scalar): scalar(scalar){
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
