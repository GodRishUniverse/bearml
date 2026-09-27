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
                T forward(T& x) override{
                    this->check_device(x);
                    auto neg_x = -1.0*x;   // named so T::exp can take it by reference
                    auto exp_node = T::exp(neg_x);
                    auto sigmoid = 1.0/(1.0 + exp_node);
                    return sigmoid;
                }

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

                T forward(T& x) override{
                    this->check_device(x);
                    T softmax_node = T::softmax(x, this->dim);
                    return softmax_node;
                }

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
                T forward(T& x) override{
                    this->check_device(x);
                    std::vector<int> temp_shape = x.getShape();
                    T temp_zero(temp_shape,this->device);
                    return T::max(x, temp_zero)+this->negative_slope*T::min(x,temp_zero);
                }

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
                T forward(T& x) override{
                    this->check_device(x);

                    // std::shared_ptr<bearml::Node<Tensor>> exp_minus = exp(-1.0*node_x);
                    // std::shared_ptr<bearml::Node<Tensor>> exp_plus = exp(node_x);

                    // auto tanh = (exp_plus - exp_minus)/(exp_plus + exp_minus); // should work now - hadamard division is supported
                    return T::tanh(x);
                }

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

                T operation(T& x) {
                    auto x_cubed = bearml::linear_algebra::hadamard(x,  bearml::linear_algebra::hadamard(x, x)) ;
                    auto inside_tanh = std::sqrt(2.0/ M_PI) * (x+ 0.044715*x_cubed);   // scalars, not T(2.0) tensors
                    auto tanh_inside = T::tanh(inside_tanh);
                    auto one_plus_tanh_x = 1.0 + tanh_inside;
                    auto res = 0.5 * bearml::linear_algebra::hadamard(x, one_plus_tanh_x);
                    return res;
                }

                // we override this from Module class
                T forward(T& x) override{
                    this->check_device(x);
                    return operation(x);
                }

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

                T operation(T& x) {

                    auto sigmoid_act = Sigmoid<T>(this->random_seed, this->device);   // same device as this layer
                    auto sigmoid_output = sigmoid_act(x);

                    return bearml::linear_algebra::hadamard(x, sigmoid_output);
                }

                // we override this from Module class
                T forward(T& x) override{
                    this->check_device(x);
                    return operation(x);
                }

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

                T operation(T& x) {
                    auto exp_x = T::exp(x);
                    auto intermediate = 1.0 + exp_x;
                    return T::log(intermediate);
                }

                // we override this from Module class
                T forward(T& x) override{
                    this->check_device(x);
                    return operation(x);
                }

                // return nothing a SoftPlus layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };

        // alias for Swish activation function
        template<typename T>
        using Swish = SiLU<T>;

    }
}

#endif
