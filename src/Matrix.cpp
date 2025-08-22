# include "Matrix.h"

// USE eigen for this  - this is not really used anywhere

// TODO : implementation needed - division (inversion -  should work for constants and matrix inversion ), GEMM

namespace simplenet{
    // matrix addition
    template<typename U>
    Matrix<U> operator+(Matrix<U> &other1,  Matrix<U> &other2) {
        if (other1.getRows() != other2.getRows() || other1.getCols() != other2.getCols()) {
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");
        }
        // saving memory by using +=
        Matrix<U> res = other1;
        res += other2;
        return res;
    }

    template<typename T>
    void Matrix<T>::operator+=( Matrix<T> &other)  {
        if (other.getRows() != this->rows || other.getCols() != this->cols) {
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this->data[i * cols + j] += other.get(i, j);
            }
        }
    }

    //copy assignment operator
    template<typename T>
    Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
        if (this != &other) {
            delete[] data;
            this->rows = other.rows;
            this->cols = other.cols;
            this->data = new T[rows * cols];
            std::copy(other.data, other.data + rows * cols, this->data);
        }
        return *this;
    }

    // move assignment operator
    template<typename T>
    Matrix<T>&& Matrix<T>::operator=( Matrix<T>&& other) {
        if (this != &other) {
            delete[] data;
            this->rows = other.rows;
            this->cols = other.cols;
            this->data = other.data;
            other.data = nullptr;
        }
        return std::move(*this);
    }

    // scalar multiplication
    template<typename T>
    Matrix<T>& Matrix<T>::operator*=(T scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this->data[i * cols + j] *= scalar;
            }
        }
        return *this;
    }


    // matrix multiplication
    template<typename T>
    Matrix<T> Matrix<T>::operator*(Matrix<T>&  mat) const {
        // TODO: Write a CUDA kernel for this
        Matrix result(rows, mat.getCols());
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < mat.getCols(); j++) {
                T sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += this->get(i, k) * mat.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    template<typename T>
    Matrix<T>::Matrix(Matrix<T>& other){
        this->rows = other.rows;
        this->cols = other.cols;
        this->data = new T[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this->data[i * cols + j] = other.get(i, j);
            }
        }
    }

    template<typename T>
    Matrix<T>::Matrix(Matrix<T> &&other) {
        this->rows = other.rows;
        this->cols = other.cols;
        std::swap(this->data, other.data);
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }


    template<typename T>
    void Matrix<T>::swapCols(int col1, int col2){
        if (col1 < 0 || col1 >= this->cols || col2 < 0 || col2 >= this->cols) {
            std::cerr << "Invalid column indices" << std::endl;
            return;
        }

        for (int i = 0; i < this->rows; i++) {
            std::swap(this->data[i * this->cols + col1], this->data[i * this->cols + col2]);
        }
    }

    template<typename T>
    void Matrix<T>::swapRows(int row1, int row2){
        if (row1 < 0 || row1 >= this->rows || row2 < 0 || row2 >= this->rows) {
            std::cerr << "Invalid row indices" << std::endl;
            return;
        }

        for (int j = 0; j < this->cols; j++) {
            std::swap(this->data[row1 * this->cols + j], this->data[row2 * this->cols + j]);
        }
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix){
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                os << matrix.get(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }

    template<typename T>
    void Matrix<T>::transpose(){
        // rows will be cols and cols will be rows
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                std::swap(this->data[i * cols + j], this->data[j * cols + i]);
            }
        }
    }

    // Xavier initialization
    template<typename T>
    static Matrix<T> xavier(int inrows, int incols, int input_size, int output_size) {

        std::random_device rd{};
        std::mt19937 gen{rd()};


        float stddev = sqrt(2.0 / (input_size + output_size));

        std::normal_distribution<double> d{0.0,stddev};

        Matrix<T> m(inrows, incols);
        for (int i = 0; i < inrows; i++) {
            for (int j = 0; j < incols; j++) {
                m.set(i, j, static_cast<T>(d(gen)));
            }
        }
        return m;
    }

    template<typename T>
    bool Matrix<T>::operator==(const Matrix<T> &m) const{
        if (this->rows != m.rows || this->cols != m.cols) {
            return false;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (this->data[i * cols + j] != m.get(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
    template<typename T>
    bool Matrix<T>::operator!=(const Matrix<T> &m) const {
        return !(*this == m);
    }

}
