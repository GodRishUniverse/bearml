#pragma once

#include <iostream>
#include <utility>
#include <ostream>
#include <random>

#ifndef MATRIX_H
#define MATRIX_H

// TODO: Templatify Matrix class so that it can work with any data type
namespace bearml{


    template<typename T> class Matrix;
    template<typename T> std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix);
  
    template <typename T>
    class Matrix
    {
    private:
        int rows;
        int cols;
        T *data; // flattened representation of the matrix

    public:
        Matrix<T>(int rows, int cols) : rows(rows), cols(cols), data(new T[rows * cols]) {
            for (int i = 0; i < rows * cols; i++) {
                this->data[i] = 0;
            }
        };

        Matrix<T>(std::initializer_list<std::initializer_list<T>> dataPassed) : rows(dataPassed.size()), cols(dataPassed.begin()->size()), data(new T[rows * cols]) {
            int i = 0;
            int j = 0;

            for (const auto& row : dataPassed) {
                for (const auto& elem : row) {
                    this->data[i * cols + j] = elem;
                    j++;
                }
                i++;
                j = 0;
            }
        };

        int getRows() const {
            return this->rows;
        }

        int getCols() const {
            return this->cols;
        }

        T get(int i, int j) const {
            return this->data[i * cols + j];
        }

        void set(int i, int j, T value) {
            this->data[i * cols + j] = value;
        }

        // transpose the matrix
        void transpose();

        // swap columns
        void swapCols(int col1, int col2);

        // swap rows
        void swapRows(int row1, int row2);

        // matrix addition
        template<typename U>
        friend Matrix<U> operator+(const Matrix<U>& other1, const Matrix<U>& other2);

        // matrix addition
        void operator+=(Matrix<T> &other) ;

        // Copy constructor
        Matrix(Matrix<T>& other);

        // Copy assignment operator
        Matrix<T>& operator=(const Matrix<T>& other);

        // Move constructor
        Matrix(Matrix<T>&& other);

        // Move assignment operator
        Matrix&& operator=( Matrix&& other);

        // scalar multiplication
        Matrix<T>& operator*=(T scalar);
        // matrix multiplication
        Matrix<T> operator*(Matrix<T>&  mat) const;
        // destructor
        ~Matrix(){delete[] data;};
        // print helper
        friend std::ostream& operator<< <T>(std::ostream& os, const Matrix<T>& matrix);

        static Matrix<T> xavier(int inrows, int incols, int input_size, int output_size);

        bool operator==(const Matrix<T> &m) const;
        bool operator!=(const Matrix<T> &m) const;
        
    };

    #endif // MATRIX_H

} 
