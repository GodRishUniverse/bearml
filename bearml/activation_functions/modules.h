#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"

using ll = long long;

#ifndef MODULES
namespace bearml {

    namespace neural_network {
        // will apply polymorphism and inheritance for activation functions as the principle idea is the same
        // will inherit the Node class for operations to be carried out and use the backward features

        // Module is generic over the tensor element type. Default stays TensorD so
        // existing TensorD-based layers keep working via Module<> / Module<TensorD>.
        template <typename T = bearml::Tensorf>
        class Module{
            protected:
                // Protected members can be called by derived classes
                int random_seed = 42;
                bearml::Device device;
                Module(int seed, bearml::Device dev) : random_seed(seed), device(dev) {}

                // input and layer must be on the same device - to() is not tracked, so moving here would cut the graph
                void check_device(const T& x) const;
            public:
                virtual ~Module()  = default; // it is a pure virtual class

                // we will assume at least one input - may change it
                virtual T forward(T& x)  =0; // pure virtual function

                virtual T operator()(T&x) =0;

                virtual void initialize_parameters() {}

                // get all the parameters here
                virtual std::vector<std::shared_ptr<T>> parameters() = 0;

                // TODO: test this
                static void xavier_init(T& t, int input_size, int output_size, int seed);

                // He initialization - to check
                static void he_init(T& t, int input_size, int seed);

        };

        // inherits operations and also gets structure from Module class
        template <typename T = bearml::Tensorf>
        class Linear : public Module<T>{
                static_assert(bearml::is_floating(bearml::dtype_of<bearml::tensor_element_t<T>>),
                    "Linear parameters must be float, double, or bfloat16 tensors");
            private:
                std::shared_ptr<T> W;  // Weight matrix as a node for autogradient
                std::shared_ptr<T> B;  // Bias vector as a node for autogradient
                int input_size;
                int output_size;
                std::string initialization_method;
            public:
                // Initialize on CPU first  so that we dont get GPU direct access errors - then transfer to the target device
                Linear(int in_shape, int out_shape, std::string initialization = "Xavier", Device dev = Device(DeviceType::CPU, 0), int random_seed =42);

                // a copy is a new layer - W/B are new deep copies made fresh leaves, tracked only if the source was
                Linear(const Linear& other);

                // copies the values into this layer's own W/B, so an optimizer holding them sees the new weights
                Linear& operator=(const Linear& other);

                // moves hand over the shared W/B - optimizers holding them stay valid
                Linear(Linear&&) = default;
                Linear& operator=(Linear&&) = default;

                T operator()(T&x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                T forward(T& x) override;

                // we will perform Xavier Init here
                void initialize_parameters() override;

                // Helpers
                T get_weights() const { return *W; }
                T get_bias() const { return *B; }

                int get_in_shape() const { return input_size; }
                int get_out_shape() const { return output_size; }

                std::vector<std::shared_ptr<T>> parameters() override{
                    return {W, B};
                }

        };

        template <typename T = bearml::Tensorf>
        class ReLU : public Module<T>{
            public:
                ReLU(int random_seed =42, Device dev =  Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                T forward(T& x) override;

                // return nothing a relu layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };

        // bodies live in modules.cpp, instantiated there for Tensorf and TensorD
        extern template class Module<bearml::Tensorf>;
        extern template class Module<bearml::TensorD>;
        extern template class Linear<bearml::Tensorf>;
        extern template class Linear<bearml::TensorD>;
        extern template class ReLU<bearml::Tensorf>;
        extern template class ReLU<bearml::TensorD>;
    }

}
#endif
