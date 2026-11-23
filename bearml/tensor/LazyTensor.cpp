#include "LazyTensor.h"
#include <stdexcept>
namespace bearml {

       // Helper function for lazy tensor graph total memory usage - TODO: understand how graph will be passed in here
    void calculate_lazy_tensor_memory_usage() {

    }

    // ------------------------------ LazyTensor ------------------------------

    LazyTensor::LazyTensor(std::shared_ptr<IRNode> node) {
        throw std::logic_error("not implemented: LazyTensor::LazyTensor");
    }

    template <typename T>
    LazyTensor LazyTensor::from(const Tensor<T>& t) {
        throw std::logic_error("not implemented: LazyTensor::from");
    }

    // needs an interpreter - leave until one exists
    template <typename T>
    Tensor<T> LazyTensor::to_tensor() const {
        throw std::logic_error("not implemented: LazyTensor::to_tensor");
    }

    const std::vector<int>& LazyTensor::shape() const {
        throw std::logic_error("not implemented: LazyTensor::shape");
    }

    DType LazyTensor::dtype() const {
        throw std::logic_error("not implemented: LazyTensor::dtype");
    }

    const Device& LazyTensor::device() const {
        throw std::logic_error("not implemented: LazyTensor::device");
    }

    const std::shared_ptr<IRNode>& LazyTensor::ir() const {
        throw std::logic_error("not implemented: LazyTensor::ir");
    }

    std::string LazyTensor::to_string() const {
        throw std::logic_error("not implemented: LazyTensor::to_string");
    }

    // ------------------------------ binary ops ------------------------------

    LazyTensor operator+(const LazyTensor& a, const LazyTensor& b) { throw std::logic_error("not implemented: operator+(LazyTensor, LazyTensor)"); }
    LazyTensor operator-(const LazyTensor& a, const LazyTensor& b) { throw std::logic_error("not implemented: operator-(LazyTensor, LazyTensor)"); }
    LazyTensor operator*(const LazyTensor& a, const LazyTensor& b) { throw std::logic_error("not implemented: operator*(LazyTensor, LazyTensor)"); }
    LazyTensor operator/(const LazyTensor& a, const LazyTensor& b) { throw std::logic_error("not implemented: operator/(LazyTensor, LazyTensor)"); }

    // ------------------------------ scalar ops ------------------------------

    LazyTensor operator+(const LazyTensor& a, double c) { throw std::logic_error("not implemented: operator+(LazyTensor, double)"); }
    LazyTensor operator+(double c, const LazyTensor& a) { throw std::logic_error("not implemented: operator+(double, LazyTensor)"); }
    LazyTensor operator-(const LazyTensor& a, double c) { throw std::logic_error("not implemented: operator-(LazyTensor, double)"); }
    LazyTensor operator-(double c, const LazyTensor& a) { throw std::logic_error("not implemented: operator-(double, LazyTensor)"); }
    LazyTensor operator*(const LazyTensor& a, double c) { throw std::logic_error("not implemented: operator*(LazyTensor, double)"); }
    LazyTensor operator*(double c, const LazyTensor& a) { throw std::logic_error("not implemented: operator*(double, LazyTensor)"); }
    LazyTensor operator/(const LazyTensor& a, double c) { throw std::logic_error("not implemented: operator/(LazyTensor, double)"); }
    LazyTensor operator/(double c, const LazyTensor& a) { throw std::logic_error("not implemented: operator/(double, LazyTensor)"); }

    // ------------------------------ unary ops ------------------------------

    LazyTensor LazyTensor::exp(const LazyTensor& a)  { throw std::logic_error("not implemented: LazyTensor::exp"); }
    LazyTensor LazyTensor::log(const LazyTensor& a)  { throw std::logic_error("not implemented: LazyTensor::log"); }
    LazyTensor LazyTensor::abs(const LazyTensor& a)  { throw std::logic_error("not implemented: LazyTensor::abs"); }
    LazyTensor LazyTensor::sqrt(const LazyTensor& a) { throw std::logic_error("not implemented: LazyTensor::sqrt"); }
    LazyTensor LazyTensor::sin(const LazyTensor& a)  { throw std::logic_error("not implemented: LazyTensor::sin"); }
    LazyTensor LazyTensor::cos(const LazyTensor& a)  { throw std::logic_error("not implemented: LazyTensor::cos"); }
    LazyTensor LazyTensor::tanh(const LazyTensor& a) { throw std::logic_error("not implemented: LazyTensor::tanh"); }

    LazyTensor LazyTensor::transpose(const LazyTensor& a) { throw std::logic_error("not implemented: LazyTensor::transpose"); }
    LazyTensor LazyTensor::sum(const LazyTensor& a, int dim) { throw std::logic_error("not implemented: LazyTensor::sum"); }
    LazyTensor LazyTensor::softmax(const LazyTensor& a, int dim) { throw std::logic_error("not implemented: LazyTensor::softmax"); }

    // explicit instantiation - from/to_tensor bodies live here, so every supported element type is built once
    template LazyTensor LazyTensor::from<float>(const Tensor<float>&);
    template LazyTensor LazyTensor::from<double>(const Tensor<double>&);
    template LazyTensor LazyTensor::from<int>(const Tensor<int>&);
    template Tensor<float> LazyTensor::to_tensor<float>() const;
    template Tensor<double> LazyTensor::to_tensor<double>() const;
    template Tensor<int> LazyTensor::to_tensor<int>() const;
    #if defined(__STDCPP_BFLOAT16_T__)
        template LazyTensor LazyTensor::from<std::bfloat16_t>(const Tensor<std::bfloat16_t>&);
        template Tensor<std::bfloat16_t> LazyTensor::to_tensor<std::bfloat16_t>() const;
    #endif
}
