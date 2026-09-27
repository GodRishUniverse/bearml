// activation layer definitions - declared in activation_functions.h, instantiated at the bottom
#include "activation_functions/activation_functions.h"

namespace bearml {
    namespace neural_network {

        template <typename T>
        T Sigmoid<T>::forward(T& x){
            this->check_device(x);
            auto neg_x = -1.0*x;   // named so T::exp can take it by reference
            auto exp_node = T::exp(neg_x);
            auto sigmoid = 1.0/(1.0 + exp_node);
            return sigmoid;
        }

        template <typename T>
        T Softmax<T>::forward(T& x){
            this->check_device(x);
            T softmax_node = T::softmax(x, this->dim);
            return softmax_node;
        }

        template <typename T>
        T LeakyReLU<T>::forward(T& x){
            this->check_device(x);
            std::vector<int> temp_shape = x.getShape();
            T temp_zero(temp_shape,this->device);
            return T::max(x, temp_zero)+this->negative_slope*T::min(x,temp_zero);
        }

        template <typename T>
        T Tanh<T>::forward(T& x){
            this->check_device(x);

            // std::shared_ptr<bearml::Node<Tensor>> exp_minus = exp(-1.0*node_x);
            // std::shared_ptr<bearml::Node<Tensor>> exp_plus = exp(node_x);

            // auto tanh = (exp_plus - exp_minus)/(exp_plus + exp_minus); // should work now - hadamard division is supported
            return T::tanh(x);
        }

        template <typename T>
        T GELU<T>::operation(T& x){
            auto x_cubed = bearml::linear_algebra::hadamard(x,  bearml::linear_algebra::hadamard(x, x)) ;
            auto inside_tanh = std::sqrt(2.0/ M_PI) * (x+ 0.044715*x_cubed);   // scalars, not T(2.0) tensors
            auto tanh_inside = T::tanh(inside_tanh);
            auto one_plus_tanh_x = 1.0 + tanh_inside;
            auto res = 0.5 * bearml::linear_algebra::hadamard(x, one_plus_tanh_x);
            return res;
        }

        template <typename T>
        T GELU<T>::forward(T& x){
            this->check_device(x);
            return operation(x);
        }

        template <typename T>
        T SiLU<T>::operation(T& x){

            auto sigmoid_act = Sigmoid<T>(this->random_seed, this->device);   // same device as this layer
            auto sigmoid_output = sigmoid_act(x);

            return bearml::linear_algebra::hadamard(x, sigmoid_output);
        }

        template <typename T>
        T SiLU<T>::forward(T& x){
            this->check_device(x);
            return operation(x);
        }

        template <typename T>
        T SoftPlus<T>::operation(T& x){
            auto exp_x = T::exp(x);
            auto intermediate = 1.0 + exp_x;
            return T::log(intermediate);
        }

        template <typename T>
        T SoftPlus<T>::forward(T& x){
            this->check_device(x);
            return operation(x);
        }

    }
}

template class bearml::neural_network::Sigmoid<bearml::Tensorf>;
template class bearml::neural_network::Sigmoid<bearml::TensorD>;
template class bearml::neural_network::Softmax<bearml::Tensorf>;
template class bearml::neural_network::Softmax<bearml::TensorD>;
template class bearml::neural_network::LeakyReLU<bearml::Tensorf>;
template class bearml::neural_network::LeakyReLU<bearml::TensorD>;
template class bearml::neural_network::Tanh<bearml::Tensorf>;
template class bearml::neural_network::Tanh<bearml::TensorD>;
template class bearml::neural_network::GELU<bearml::Tensorf>;
template class bearml::neural_network::GELU<bearml::TensorD>;
template class bearml::neural_network::SiLU<bearml::Tensorf>;
template class bearml::neural_network::SiLU<bearml::TensorD>;
template class bearml::neural_network::SoftPlus<bearml::Tensorf>;
template class bearml::neural_network::SoftPlus<bearml::TensorD>;
