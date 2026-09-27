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
                void check_device(const T& x) const {
                    if (x.getDevice() != this->device) {
                        throw std::invalid_argument("input tensor and layer are on different devices");
                    }
                }
            public:
                virtual ~Module()  = default; // it is a pure virtual class

                // we will assume at least one input - may change it
                virtual T forward(T& x)  =0; // pure virtual function

                virtual T operator()(T&x) =0;

                virtual void initialize_parameters() {}

                // get all the parameters here
                virtual std::vector<std::shared_ptr<T>> parameters() = 0;

                // TODO: test this
                static void xavier_init(T& t, int input_size, int output_size, int seed) {

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
                static void he_init(T& t, int input_size, int seed){
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
                Linear(int in_shape, int out_shape, std::string initialization = "Xavier", Device dev = Device(DeviceType::CPU, 0), int random_seed =42) : Module<T>(random_seed, dev), W(std::make_shared<T>(std::vector<int>{in_shape, out_shape})), B(std::make_shared<T>(std::vector<int>{out_shape})), input_size(in_shape), output_size(out_shape), initialization_method(initialization){
                    initialize_parameters();

                    if (!dev.is_cpu()) {
                        W->to_(dev);
                        B->to_(dev);
                    }

                    // tracked only after the values and device are final
                    W->set_requires_grad();
                    B->set_requires_grad();
                }

                // a copy is a new layer - W/B are new deep copies made fresh leaves, tracked only if the source was
                Linear(const Linear& other) : Module<T>(other), W(std::make_shared<T>(*other.W)), B(std::make_shared<T>(*other.B)), input_size(other.input_size), output_size(other.output_size), initialization_method(other.initialization_method){
                    W->set_requires_grad(other.W->requires_grad());
                    B->set_requires_grad(other.B->requires_grad());
                }

                // copies the values into this layer's own W/B, so an optimizer holding them sees the new weights
                Linear& operator=(const Linear& other){
                    if (this != &other) {
                        Module<T>::operator=(other);
                        *W = *other.W;
                        *B = *other.B;
                        input_size = other.input_size;
                        output_size = other.output_size;
                        initialization_method = other.initialization_method;
                        W->set_requires_grad(other.W->requires_grad());
                        B->set_requires_grad(other.B->requires_grad());
                    }
                    return *this;
                }

                // moves hand over the shared W/B - optimizers holding them stay valid
                Linear(Linear&&) = default;
                Linear& operator=(Linear&&) = default;

                T operator()(T&x) override {
                    return this->forward(x);
                }

                // we override this from Module class
                T forward(T& x) override{
                    this->check_device(x);
                    return x * (*W) + (*B); // convert input shape to output shape
                }

                // we will perform Xavier Init here
                void initialize_parameters() override{
                    // default initialization is always 0 as we already know as that is what our Tensor class does
                    if (this->initialization_method == "Zero") {
                        // default behaviour
                    }
                    else if (this->initialization_method == "Xavier"){
                        this->xavier_init(*W, input_size, output_size, this->random_seed);
                    } else if (this->initialization_method == "He"){
                        this->he_init(*W, input_size, this->random_seed);
                    } else {
                        throw std::invalid_argument("Invalid initialization method");
                    }
                    // we dont initialize the bias Tensor at the moment
                };

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
                T forward(T& x) override{
                    this->check_device(x);
                    std::vector<int> temp_shape = x.getShape();
                    T temp_zero(temp_shape,this->device);
                    return T::max(x, temp_zero);
                }

                // return nothing a relu layer does not have parameters
                std::vector<std::shared_ptr<T>> parameters() override{
                    return {};
                }

        };
    }

}
#endif
