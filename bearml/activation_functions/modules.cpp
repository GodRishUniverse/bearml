// Module, Linear and ReLU definitions - declared in modules.h, instantiated at the bottom
#include "activation_functions/modules.h"

namespace bearml {
    namespace neural_network {

        template <typename T>
        void Module<T>::check_device(const T& x) const{
            if (x.getDevice() != this->device) {
                throw std::invalid_argument("input tensor and layer are on different devices");
            }
        }

        template <typename T>
        void Module<T>::xavier_init(T& t, int input_size, int output_size, int seed){

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

        template <typename T>
        void Module<T>::he_init(T& t, int input_size, int seed){
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

        template <typename T>
        Linear<T>::Linear(int in_shape, int out_shape, std::string initialization, Device dev, int random_seed) : Module<T>(random_seed, dev), W(std::make_shared<T>(std::vector<int>{in_shape, out_shape})), B(std::make_shared<T>(std::vector<int>{out_shape})), input_size(in_shape), output_size(out_shape), initialization_method(initialization){
            initialize_parameters();

            if (!dev.is_cpu()) {
                W->to_(dev);
                B->to_(dev);
            }

            // tracked only after the values and device are final
            W->set_requires_grad();
            B->set_requires_grad();
        }

        template <typename T>
        Linear<T>::Linear(const Linear& other) : Module<T>(other), W(std::make_shared<T>(*other.W)), B(std::make_shared<T>(*other.B)), input_size(other.input_size), output_size(other.output_size), initialization_method(other.initialization_method){
            W->set_requires_grad(other.W->requires_grad());
            B->set_requires_grad(other.B->requires_grad());
        }

        template <typename T>
        Linear<T>& Linear<T>::operator=(const Linear& other){
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

        template <typename T>
        T Linear<T>::forward(T& x){
            this->check_device(x);
            return x * (*W) + (*B); // convert input shape to output shape
        }

        template <typename T>
        void Linear<T>::initialize_parameters(){
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

        template <typename T>
        T ReLU<T>::forward(T& x){
            this->check_device(x);
            std::vector<int> temp_shape = x.getShape();
            T temp_zero(temp_shape,this->device);
            return T::max(x, temp_zero);
        }

    }
}

template class bearml::neural_network::Module<bearml::Tensorf>;
template class bearml::neural_network::Module<bearml::TensorD>;
template class bearml::neural_network::Linear<bearml::Tensorf>;
template class bearml::neural_network::Linear<bearml::TensorD>;
template class bearml::neural_network::ReLU<bearml::Tensorf>;
template class bearml::neural_network::ReLU<bearml::TensorD>;
