#pragma once
#include "tensor/Tensor.h"
#include <iostream>
#include <string>
#include <optional>
#include <vector>
#include "utils.h"
// we wrill write our dataset class here
#ifndef BEARML_DATASETS_H
#define BEARML_DATASETS_H

namespace bearml {
    namespace data {

        template<typename T>
        class  Sample {
            public:
                bearml::Tensor<T> input; // x
                bearml::Tensor<T> output; //y
        };

        // will be abstract class as Users will have to implement their own datasets
        template <typename T>
        class Dataset {
        public:
            virtual ~Dataset() = default; // default destructor
            // returns the number of samples in the dataset
            int64_t get_length() const =0; // pure virtual

            virtual int64_t size() const = 0;
            virtual Sample<T> get(int64_t index) const = 0;

            // returns the data at the given index
            Sample<T> operator[](int64_t i) const { return get(i); }
        };

    }
}

#endif
