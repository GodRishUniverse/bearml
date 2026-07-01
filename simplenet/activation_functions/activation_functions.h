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
namespace simplenet {

    namespace neural_network {


        // inherits from module -> need to test it
        template <typename T = simplenet::Tensorf>
        class Sigmoid : public Module<T>{
            public:
                Sigmoid(int random_seed =42, Device dev = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev){

                }

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }
                    std::shared_ptr<simplenet::Node<T>> node_x = exp(-1.0*x);
                    auto sigmoid = 1.0/(1.0 + node_x);
                    return sigmoid;
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }
                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    auto exp_node = exp(-1.0*node_x);
                    auto sigmoid = 1.0/(1.0 + exp_node);
                    return sigmoid;
                }

                T get_detached_value(T& t)override {
                    return this->forward(t)->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a sigmoid layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };


        template <typename T = simplenet::Tensorf>
        class Softmax : public Module<T>{

            int dim;
            public:
                Softmax(int dim, int random_seed =42, Device dev = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev), dim(dim){

                }

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }
                    std::shared_ptr<simplenet::Node<T>> softmax_node = softmax(x, this->dim);
                    return softmax_node;
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }
                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    std::shared_ptr<simplenet::Node<T>>  softmax_node = softmax(node_x, this->dim);
                    return softmax_node;
                }

                T get_detached_value(T& t)override {
                    return this->forward(t)->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a softmax layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };



        // leaky relu
        template <typename T = simplenet::Tensorf>
        class LeakyReLU : public Module<T>{
            private:
                double negative_slope;

            public:
                LeakyReLU(double negative_slope = 0.01, int random_seed = 42, Device dev = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, dev), negative_slope(negative_slope) {}

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    std::vector<int> temp_shape = x->val.getShape();
                    T temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<T>> mask_node = simplenet::Node<T>::make_node(temp_zero);
                    return max(x, mask_node)+this->negative_slope*min(x,mask_node);
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    std::vector<int> temp_shape = x.getShape();
                    T temp_zero(temp_shape,this->device);
                    std::shared_ptr<simplenet::Node<T>> mask_node = simplenet::Node<T>::make_node(temp_zero);
                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    return max(node_x, mask_node)+this->negative_slope*min(node_x,mask_node);
                }

                T get_detached_value(T& t)override {
                    std::vector<int> temp_shape = t.getShape();
                    T temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<T>> mask_node = simplenet::Node<T>::make_node(temp_zero);
                    std::shared_ptr<simplenet::Node<T>> node_t = simplenet::Node<T>::make_node(t);
                    return (max(node_t, mask_node)+this->negative_slope*min(node_t,mask_node))->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    std::vector<int> temp_shape = t->val.getShape();
                    T temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<T>> mask_node = simplenet::Node<T>::make_node(temp_zero);
                    return (max(t, mask_node)+this->negative_slope*min(t,mask_node))->val;
                }

                // return nothing a relu layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        template <typename T = simplenet::Tensorf>
        class Tanh : public Module<T>{
            public:
                Tanh(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }

                    return tanh(x);
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }

                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    // std::shared_ptr<simplenet::Node<Tensor>> exp_minus = exp(-1.0*node_x);
                    // std::shared_ptr<simplenet::Node<Tensor>> exp_plus = exp(node_x);

                    // auto tanh = (exp_plus - exp_minus)/(exp_plus + exp_minus); // should work now - hadamard division is supported
                    return tanh(node_x);
                }

                T get_detached_value(T& t)override {
                    return this->forward(t)->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a tanh layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        template <typename T = simplenet::Tensorf>
        class GELU : public Module<T>{
            public:
                GELU(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operation(std::shared_ptr<simplenet::Node<T>> x) {
                    auto x_cubed = simplenet::Node<T>::hadamard(x,  simplenet::Node<T>::hadamard(x, x)) ;
                    auto inside_tanh = sqrt(T(2.0)/ T(M_PI)) * (x+ T(0.044715)*x_cubed);
                    auto one_plus_tanh_x = T(1.0) + tanh(inside_tanh);
                    auto res = T(0.5) * simplenet::Node<T>::hadamard(x, one_plus_tanh_x);
                    return res;
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }
                    return operation(x);
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }

                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    return operation(node_x);
                }

                T get_detached_value(T& t)override {
                    return this->forward(t)->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a GELU layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };



        // inherits from module -> need to test it
        template <typename T = simplenet::Tensorf>
        class SiLU : public Module<T>{
            public:
                SiLU(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operation(std::shared_ptr<simplenet::Node<T>> x) {

                    auto sigmoid_act = Sigmoid<T>();
                    auto sigmoid_output = sigmoid_act(x);

                    return simplenet::Node<T>::hadamard(x, sigmoid_output);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }
                    return operation(x);
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }

                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    return operation(node_x);
                }

                T get_detached_value(T& t)override {
                    return this->forward(t)->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a SiLU layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        template <typename T = simplenet::Tensorf>
        class SoftPlus : public Module<T>{
            public:
                SoftPlus(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module<T>(random_seed, device){

                }

                std::shared_ptr<simplenet::Node<T>> operator()(T&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operator()(std::shared_ptr<simplenet::Node<T>> x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<T>> operation(std::shared_ptr<simplenet::Node<T>> x) {
                    auto intermediate = T(1.0) + simplenet::Node<T>::exp(x);
                    return simplenet::Node<T>::log(intermediate);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<T>> forward(std::shared_ptr<simplenet::Node<T>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }
                    return operation(x);
                }

                std::shared_ptr<simplenet::Node<T>> forward(T& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }

                    std::shared_ptr<simplenet::Node<T>> node_x = simplenet::Node<T>::make_node(x);
                    return operation(node_x);
                }

                T get_detached_value(T& t)override {
                    return this->forward(t)->val;
                }

                T get_detached_value(std::shared_ptr<simplenet::Node<T>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a SoftPlus layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<T>>> parameters() override{
                    return {};
                }

        };

        // alias for Swish activation function
        template<typename T>
        using Swish = SiLU<T>;

    }
}

#endif
