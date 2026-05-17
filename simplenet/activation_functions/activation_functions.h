#pragma once
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"
#include "modules.h"

#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS
namespace simplenet {

    namespace neural_network {


        // inherits from module -> need to test it
        class Sigmoid : public Module{
            public:
                Sigmoid(int random_seed =42, Device dev = Device(DeviceType::CPU, 0)) : Module(random_seed, dev){

                }

                std::shared_ptr<simplenet::Node<TensorD>> operator()(TensorD&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<TensorD>> operator()(std::shared_ptr<simplenet::Node<TensorD>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<TensorD>> forward(std::shared_ptr<simplenet::Node<TensorD>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }
                    std::shared_ptr<simplenet::Node<TensorD>> node_x = exp(-1.0*x);
                    auto sigmoid = 1.0/(1.0 + node_x);
                    return sigmoid;
                }

                std::shared_ptr<simplenet::Node<TensorD>> forward(TensorD& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }
                    std::shared_ptr<simplenet::Node<TensorD>> node_x = simplenet::Node<simplenet::TensorD>::make_node(x);
                    auto exp_node = exp(-1.0*node_x);
                    auto sigmoid = 1.0/(1.0 + exp_node);
                    return sigmoid;
                }

                TensorD get_detached_value(TensorD& t)override {
                    return this->forward(t)->val;
                }

                TensorD get_detached_value(std::shared_ptr<simplenet::Node<TensorD>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a sigmoid layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> parameters() override{
                    return {};
                }

        };


        // leaky relu
        class LeakyReLU : public Module{
            private:
                double negative_slope;

            public:
                LeakyReLU(double negative_slope = 0.01, int random_seed = 42, Device dev = Device(DeviceType::CPU, 0)) : Module(random_seed, dev), negative_slope(negative_slope) {}

                std::shared_ptr<simplenet::Node<TensorD>> operator()(TensorD&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<TensorD>> operator()(std::shared_ptr<simplenet::Node<TensorD>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<TensorD>> forward(std::shared_ptr<simplenet::Node<TensorD>> x) override{
                    std::vector<int> temp_shape = x->val.getShape();
                    TensorD temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<TensorD>> mask_node = simplenet::Node<simplenet::TensorD>::make_node(temp_zero);
                    return max(x, mask_node)+this->negative_slope*min(x,mask_node);
                }

                std::shared_ptr<simplenet::Node<TensorD>> forward(TensorD& x) override{
                    std::vector<int> temp_shape = x.getShape();
                    TensorD temp_zero(temp_shape,this->device);
                    std::shared_ptr<simplenet::Node<TensorD>> mask_node = simplenet::Node<simplenet::TensorD>::make_node(temp_zero);
                    std::shared_ptr<simplenet::Node<TensorD>> node_x = simplenet::Node<simplenet::TensorD>::make_node(x);
                    return max(node_x, mask_node)+this->negative_slope*min(node_x,mask_node);
                }

                TensorD get_detached_value(TensorD& t)override {
                    std::vector<int> temp_shape = t.getShape();
                    TensorD temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<TensorD>> mask_node = simplenet::Node<simplenet::TensorD>::make_node(temp_zero);
                    std::shared_ptr<simplenet::Node<TensorD>> node_t = simplenet::Node<simplenet::TensorD>::make_node(t);
                    return (max(node_t, mask_node)+this->negative_slope*min(node_t,mask_node))->val;
                }

                TensorD get_detached_value(std::shared_ptr<simplenet::Node<TensorD>> t)override {
                    std::vector<int> temp_shape = t->val.getShape();
                    TensorD temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<TensorD>> mask_node = simplenet::Node<simplenet::TensorD>::make_node(temp_zero);
                    return (max(t, mask_node)+this->negative_slope*min(t,mask_node))->val;
                }

                // return nothing a relu layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        class Tanh : public Module{
            public:
                Tanh(int random_seed =42, Device device = Device(DeviceType::CPU, 0)) : Module(random_seed, device){

                }

                std::shared_ptr<simplenet::Node<TensorD>> operator()(TensorD&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<TensorD>> operator()(std::shared_ptr<simplenet::Node<TensorD>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<TensorD>> forward(std::shared_ptr<simplenet::Node<TensorD>> x) override{
                    if (x->val.getDevice() != this->device) {
                        x->val = x->val.to(this->device); // move to device if not already on it
                    }

                    return tanh(x);
                }

                std::shared_ptr<simplenet::Node<TensorD>> forward(TensorD& x) override{
                    if (x.getDevice() != this->device) {
                        x = x.to(this->device); // move to device if not already on it
                    }

                    std::shared_ptr<simplenet::Node<TensorD>> node_x = simplenet::Node<simplenet::TensorD>::make_node(x);
                    // std::shared_ptr<simplenet::Node<Tensor>> exp_minus = exp(-1.0*node_x);
                    // std::shared_ptr<simplenet::Node<Tensor>> exp_plus = exp(node_x);

                    // auto tanh = (exp_plus - exp_minus)/(exp_plus + exp_minus); // should work now - hadamard division is supported
                    return tanh(node_x);
                }

                TensorD get_detached_value(TensorD& t)override {
                    return this->forward(t)->val;
                }

                TensorD get_detached_value(std::shared_ptr<simplenet::Node<TensorD>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a tanh layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> parameters() override{
                    return {};
                }

        };

    }
}

#endif
