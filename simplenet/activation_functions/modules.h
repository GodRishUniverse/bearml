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
namespace simplenet {

    namespace neural_network {
        // will apply polymorphism and inheritance for activation functions as the principle idea is the same
        // will inherit the Node class for operations to be carried out and use the backward features

        class Module{
            protected:
                // Protected members can be called by derived classes
                int random_seed = 42;
                simplenet::Device device;
                Module(int seed, simplenet::Device dev) : random_seed(seed), device(dev) {}
            public:
                virtual ~Module()  = default; // it is a pure virtual class

                // we will assume at least one input - may change it
                virtual std::shared_ptr<simplenet::Node<Tensor>> forward(std::shared_ptr<simplenet::Node<Tensor>> x) = 0; // pure virtual function
                virtual std::shared_ptr<simplenet::Node<Tensor>> forward(Tensor& x)  =0; // another virtual representation

                virtual std::shared_ptr<simplenet::Node<Tensor>> operator()(Tensor&x) =0;
                virtual std::shared_ptr<simplenet::Node<Tensor>> operator()(std::shared_ptr<simplenet::Node<Tensor>> x) =0;

                virtual Tensor get_detached_value(Tensor& t) = 0;
                virtual Tensor get_detached_value(std::shared_ptr<simplenet::Node<Tensor>> t) = 0;

                virtual void initialize_parameters() {}

                // get all the parameters here
                virtual std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> parameters() = 0;

                // TODO: test this
                static void xavier_init(Tensor& t, int input_size, int output_size, int seed) {

                    // std::random_device rd{};
                    std::mt19937 gen(seed);


                    float stddev = sqrt(2.0 / (input_size + output_size));

                    std::normal_distribution<double> d{0.0,stddev};

                    // Case 1: vector or scalar
                    if (t.getShape().size()==1){
                        for (int i = 0; i < t.getShape()[0]; i++) {
                            t.set_with_offset(0, 0,i, d(gen));
                        }
                    }
                    // Case 2: matrix
                    else if (t.getShape().size()>=2){
                        ll batches = t.sizeOfTensor();
                        int rows = t.getShape()[t.getShape().size()-2];
                        int cols = t.getShape()[t.getShape().size()-1];
                        batches/= ( rows*cols );

                        for (ll b = 0; b <batches; b++){
                            for (int i = 0; i < rows; i++) {
                                for (int j = 0; j < cols; j++) {
                                    t.set_with_offset(b, i,j, d(gen));
                                }
                            }
                        }
                    }else{
                        std::invalid_argument("Not a proper Shape -> Should not reach here");
                    }
                }

                // He initialization - to check
                static void he_init(Tensor& t, int input_size, int seed){
                    // std::random_device rd{};
                    std::mt19937 gen(seed);

                    float stddev = sqrt(2.0 / (input_size)); // He initialization factor

                    std::normal_distribution<double> d{0.0,stddev};

                    // Case 1: vector or scalar
                    if (t.getShape().size()==1){
                        for (int i = 0; i < t.getShape()[0]; i++) {
                            t.set_with_offset(0, 0,i, d(gen));
                        }
                    }
                    // Case 2: matrix
                    else if (t.getShape().size()>=2){
                        ll batches = t.sizeOfTensor();
                        int rows = t.getShape()[t.getShape().size()-2];
                        int cols = t.getShape()[t.getShape().size()-1];
                        batches/= ( rows*cols );

                        for (ll b = 0; b <batches; b++){
                            for (int i = 0; i < rows; i++) {
                                for (int j = 0; j < cols; j++) {
                                    t.set_with_offset(b, i,j, d(gen));
                                }
                            }
                        }
                    }else{
                        std::invalid_argument("Not a proper Shape -> Should not reach here");
                    }
                }

        };

        // inherits operations and also gets structure from Module class
        class Linear : public Module{
            private:
                std::shared_ptr<simplenet::Node<Tensor>> W;  // Weight matrix as a node for autogradient
                std::shared_ptr<simplenet::Node<Tensor>> B;  // Bias vector as a node for autogradient
                int input_size;
                int output_size;
                std::string initialization_method;
            public:
                Linear(int in_shape, int out_shape, std::string initialization = "Xavier", Device dev = Device(DeviceType::CPU, 0), int random_seed =42) : Module(random_seed, dev), input_size(in_shape), output_size(out_shape), initialization_method(initialization){
                    // Initialize on CPU first  so that we dont get GPU direct access errors - then transfer to the target device
                    Tensor weight_tensor({input_size, output_size});
                    Tensor bias_tensor({ output_size});

                    W = simplenet::Node<Tensor>::make_node(weight_tensor);
                    B = simplenet::Node<Tensor>::make_node(bias_tensor);

                    initialize_parameters();

                    if (!dev.is_cpu()) {
                        W->val.to_(dev);
                        B->val.to_(dev);
                    }
                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(Tensor&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(std::shared_ptr<simplenet::Node<Tensor>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<Tensor>> forward(std::shared_ptr<simplenet::Node<Tensor>> x) override{
                    return x * W + B;
                }

                std::shared_ptr<simplenet::Node<Tensor>> forward(Tensor& x) override{
                    std::shared_ptr<simplenet::Node<Tensor>> node_x = simplenet::Node<simplenet::Tensor>::make_node(x);
                    return node_x * W + B; // convert input shape to output shape
                }

                Tensor get_detached_value(Tensor& t)override {
                    // t.printShape();
                    // W->val.printShape();
                    // B->val.printShape();
                    return (t*W->val + B->val);
                }

                Tensor get_detached_value(std::shared_ptr<simplenet::Node<Tensor>> t)override {
                    // t.printShape();
                    // W->val.printShape();
                    // B->val.printShape();
                    return (t->val*W->val + B->val);
                }

                // we will perform Xavier Init here
                void initialize_parameters() override{
                    // default initialization is always 0 as we already know as that is what our Tensor class does
                    if (this->initialization_method == "Zero") {
                        // default behaviour
                    }
                    else if (this->initialization_method == "Xavier"){
                        this->xavier_init(W->val, input_size, output_size, this->random_seed);
                    } else if (this->initialization_method == "He"){
                        this->he_init(W->val, input_size, this->random_seed);
                    } else {
                        throw std::invalid_argument("Invalid initialization method");
                    }
                    // we dont initialize the bias Tensor at the moment
                };

                // Helpers
                Tensor get_weights() const { return W->val; }
                Tensor get_bias() const { return B->val; }

                int get_in_shape() const { return input_size; }
                int get_out_shape() const { return output_size; }

                std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> parameters() override{
                    return {W,B};
                }

        };

        class ReLU : public Module{
            public:
                ReLU(int random_seed =42, Device dev =  Device(DeviceType::CPU, 0)) : Module(random_seed, dev){

                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(Tensor&x) override {
                    return this->forward(x);
                }

                std::shared_ptr<simplenet::Node<Tensor>> operator()(std::shared_ptr<simplenet::Node<Tensor>> x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                std::shared_ptr<simplenet::Node<Tensor>> forward(std::shared_ptr<simplenet::Node<Tensor>> x) override{
                    std::vector<int> temp_shape = x->val.getShape();
                    Tensor temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<Tensor>> mask_node = simplenet::Node<simplenet::Tensor>::make_node(temp_zero);
                    return max(x, mask_node);
                }

                std::shared_ptr<simplenet::Node<Tensor>> forward(Tensor& x) override{
                    std::vector<int> temp_shape = x.getShape();
                    Tensor temp_zero(temp_shape,this->device);
                    std::shared_ptr<simplenet::Node<Tensor>> mask_node = simplenet::Node<simplenet::Tensor>::make_node(temp_zero);
                    std::shared_ptr<simplenet::Node<Tensor>> node_x = simplenet::Node<simplenet::Tensor>::make_node(x);
                    return max(node_x, mask_node);
                }

                Tensor get_detached_value(Tensor& t)override {
                    std::vector<int> temp_shape = t.getShape();
                    Tensor temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<Tensor>> mask_node = simplenet::Node<simplenet::Tensor>::make_node(temp_zero);
                    std::shared_ptr<simplenet::Node<Tensor>> node_t = simplenet::Node<simplenet::Tensor>::make_node(t);
                    return (max(node_t, mask_node))->val;
                }

                Tensor get_detached_value(std::shared_ptr<simplenet::Node<Tensor>> t)override {
                    std::vector<int> temp_shape = t->val.getShape();
                    Tensor temp_zero(temp_shape, this->device);
                    std::shared_ptr<simplenet::Node<Tensor>> mask_node = simplenet::Node<simplenet::Tensor>::make_node(temp_zero);
                    return (max(t, mask_node))->val;
                }

                // return nothing a relu layer does not have parameters
                std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensor>>> parameters() override{
                    return {};
                }

        };
    }

}
#endif
