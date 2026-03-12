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
                Sigmoid(int random_seed =42) : Module(random_seed){

                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(Tensor&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(std::shared_ptr<simplenet::Node<Tensor>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<Tensor>> forward(std::shared_ptr<simplenet::Node<Tensor>> x) override{
                    std::shared_ptr<simplenet::Node<Tensor>> node_x = exp(-1.0*x);
                    auto sigmoid = 1.0/(1.0 + node_x);
                    return sigmoid;
                }

                std::shared_ptr<simplenet::Node<Tensor>> forward(Tensor& x) override{

                    std::shared_ptr<simplenet::Node<Tensor>> node_x = simplenet::Node<simplenet::Tensor>::make_node(x);
                    auto exp_node = exp(-1.0*node_x);
                    auto sigmoid = 1.0/(1.0 + exp_node);
                    return sigmoid;
                }

                Tensor get_detached_value(Tensor& t)override {
                    return this->forward(t)->val;
                }

                Tensor get_detached_value(std::shared_ptr<simplenet::Node<Tensor>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a sigmoid layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> parameters() override{
                    return {};
                }

        };


        // inherits from module -> need to test it
        class Tanh : public Module{
            public:
                Tanh(int random_seed =42) : Module(random_seed){

                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(Tensor&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(std::shared_ptr<simplenet::Node<Tensor>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<Tensor>> forward(std::shared_ptr<simplenet::Node<Tensor>> x) override{
                    std::shared_ptr<simplenet::Node<Tensor>> exp_minus = exp(-1.0*x);
                    std::shared_ptr<simplenet::Node<Tensor>> exp_plus = exp(x);

                    auto tanh = (exp_plus - exp_minus)/(exp_plus + exp_minus); // this will break TODO: fix cause operator/ is not implemented for Node pointers and Tensor
                    return tanh;
                }

                std::shared_ptr<simplenet::Node<Tensor>> forward(Tensor& x) override{
                    std::shared_ptr<simplenet::Node<Tensor>> node_x = simplenet::Node<simplenet::Tensor>::make_node(x);

                    std::shared_ptr<simplenet::Node<Tensor>> exp_minus = exp(-1.0*node_x);
                    std::shared_ptr<simplenet::Node<Tensor>> exp_plus = exp(node_x);

                    auto tanh = (exp_plus - exp_minus)/(exp_plus + exp_minus); // this will break TODO: fix cause operator/ is not implemented for Node pointers and Tensor
                    return tanh;
                }

                Tensor get_detached_value(Tensor& t)override {
                    return this->forward(t)->val;
                }

                Tensor get_detached_value(std::shared_ptr<simplenet::Node<Tensor>> t)override {
                    return this->forward(t)->val;
                }

                // return nothing a tanh layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> parameters() override{
                    return {};
                }

        };

    }
}

#endif
