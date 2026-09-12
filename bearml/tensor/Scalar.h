#pragma once

namespace bearml {

    template<typename T>
    class Scalar {
        private:
            T scalar;
        public:
            Scalar<T>(T scalar): scalar(scalar){
            }

            Scalar(const Scalar<T>& scalar) = default;
            Scalar& operator=(const Scalar<T>& other) = default;
            Scalar(const Scalar<T>&& scalar) = default;
            Scalar& operator=(const Scalar<T>&& other) = default;
            ~Scalar<T>() = default;

            // inherit ops
    };
}
