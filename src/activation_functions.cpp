#include "Tensor.h"
#include "Matrix.h"


#include <iostream>
#include <algorithm>

// TODO: edit according to autogradient and Node implementations in the autogradient.h/.cpp files
// TODO: make tensor activation functions

// TODO: Just my 2 cents so far - activation functions might be a class that inherit the Node class as the functionality needs to be inherited and grad values need to be specified
namespace simplenet::activation_functions{

    const double MY_PI = 3.141592653589793;

    // Linear
    template<typename T>
    Matrix<T> linear(Matrix<T> &m1, int outrows) {
        // need a way to initialize the weight matrix here.
        int inrows = m1.getRows();
        int incols = m1.getCols();

        // outcols = inrows
        int outcols = inrows;

        Matrix<T> weight= Matrix<T>::xavier(outrows, inrows, inrows * incols, outrows * outcols); // get the weight matrix
        Matrix<T> bias = Matrix<T>::xavier(outrows, inrows, inrows * incols, outrows * outcols); // get the bias

        Matrix<T> result = (weight * m1) + bias;
        return result;
    }

    // Sigmoid
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Matrix specialized
    template<typename T>
    Matrix<T> sigmoid(Matrix<T> &m) {
        Matrix<T> result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result.set(sigmoid(m.get(i, j)),i,j);
            }
        }
    }


    // ReLU
    double relu(double x) {
        return std::max(0.0, x);
    }

    template<typename T>
    Matrix<T> relu(Matrix<T> &m) {
        Matrix<T> result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result.set(std::max(0.0, m.get(i, j)), i, j);
            }
        }
    }


    // Leaky ReLU
    double leaky_relu(double x, double leaky) {
        return std::max(leaky * x, x);
    }

    template<typename T>
    Matrix<T> leaky_relu(Matrix<T> &m, double leaky) {
        Matrix<T> result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result.set(std::max(leaky * m.get(i, j), m.get(i, j)), i, j);
            }
        }
    }

    // SiLU
    double silu(double x) {
        return x / (1.0 + exp(-x));
    }

    template<typename T>
    Matrix<T> silu(Matrix<T> &m) {
        Matrix<T> result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result.set(silu(m.get(i, j)), i, j);
            }
        }
    }

    // Tanh
    double tanh(double x) {
        return (exp(2*x)-1)/(exp(2*x)+1);
    }

    template<typename T>
    Matrix<T> tanh(Matrix<T> &m) {
        Matrix<T> result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result.set(tanh(m.get(i, j)), i, j);
            }
        }
    }

    //gelu GELU(x)=0.5∗x∗(1+Tanh( sqrt(2/π)(x+0.044715∗x^3))

    double gelu(double x) {
        return 0.5 * x * (1 + tanh(sqrt(2 / MY_PI) * (x + 0.044715 * pow(x, 3))));
    }

    template<typename T>
    Matrix<T> gelu(Matrix<T> &m) {
        Matrix<T> result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result.set(gelu(m.get(i, j)), i, j);
            }
        }
    }


}

// Tensor implementation to be added
