#pragma once
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <math.h>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"
#include "modules.h"

#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS
namespace bearml {

    namespace neural_network {


        // inherits from module -> need to test it
        template <typename T = bearml::Tensorf>
        class Sigmoid : public Module<T>{
            public:
                Sigmoid(int random_seed =42, Device dev = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                T forward(T& x) override;

                // return nothing a sigmoid layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };


        template <typename T = bearml::Tensorf>
        class Softmax : public Module<T>{

            int dim;
            public:
                Softmax(int dim, int random_seed =42, Device dev = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev), dim(dim){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                T forward(T& x) override;

                // return nothing a softmax layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };



        // leaky relu
        template <typename T = bearml::Tensorf>
        class LeakyReLU : public Module<T>{
            private:
                double negative_slope;

            public:
                LeakyReLU(double negative_slope = 0.01, int random_seed = 42, Device dev = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev), negative_slope(negative_slope) {}

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


        // inherits from module -> need to test it
        template <typename T = bearml::Tensorf>
        class Tanh : public Module<T>{
            public:
                Tanh(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                T forward(T& x) override;

                // return nothing a tanh layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        template <typename T = bearml::Tensorf>
        class GELU : public Module<T>{
            public:
                GELU(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                T operation(T& x);

                // we override this from Module class
                T forward(T& x) override;

                // return nothing a GELU layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };



        // inherits from module -> need to test it
        template <typename T = bearml::Tensorf>
        class SiLU : public Module<T>{
            public:
                SiLU(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                T operation(T& x);

                // we override this from Module class
                T forward(T& x) override;

                // return nothing a SiLU layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        template <typename T = bearml::Tensorf>
        class SoftPlus : public Module<T>{
            public:
                SoftPlus(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                T operator()(T&x) override {
                    return this->forward(x);
                }

                T operation(T& x);

                // we override this from Module class
                T forward(T& x) override;

                // return nothing a SoftPlus layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };

        // alias for Swish activation function
        template<typename T>
        using Swish = SiLU<T>;

        // bodies live in activation_functions.cpp, instantiated there for Tensorf and TensorD
        extern template class Sigmoid<bearml::Tensorf>;
        extern template class Sigmoid<bearml::TensorD>;
        extern template class Softmax<bearml::Tensorf>;
        extern template class Softmax<bearml::TensorD>;
        extern template class LeakyReLU<bearml::Tensorf>;
        extern template class LeakyReLU<bearml::TensorD>;
        extern template class Tanh<bearml::Tensorf>;
        extern template class Tanh<bearml::TensorD>;
        extern template class GELU<bearml::Tensorf>;
        extern template class GELU<bearml::TensorD>;
        extern template class SiLU<bearml::Tensorf>;
        extern template class SiLU<bearml::TensorD>;
        extern template class SoftPlus<bearml::Tensorf>;
        extern template class SoftPlus<bearml::TensorD>;

    }
}

#endif
