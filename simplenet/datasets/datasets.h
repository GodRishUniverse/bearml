#pragma once
#include "tensor/Tensor.h"
#include <iostream>
#include <string>
#include <optional>
#include <vector>
// we wrill write our dataset class here
#ifndef SIMPLENET_DATASETS_H
#define SIMPLENET_DATASETS_H

namespace simplenet {
    namespace data {

        // will be abstract class as Users will have to implement their own datasets
        template <typename T>
        class Dataset {
            std::string folder_path; // path to the folder containing the dataset
            std::optional<int64_t> length; // number of samples in the dataset
            std::vector<simplenet::Tensor<T>> data; // the data in the dataset
        public:
            virtual ~Dataset() = default; // default destructor

            Dataset(const std::string& folder_path) : folder_path(folder_path) {} // constructor

            // returns the number of samples in the dataset
            int64_t get_length() const {
                if (length)
                    return *length;
                else
                    throw std::runtime_error("Length not set"); // This is needed by dataloader
            }

            // returns the data at the given index
            const simplenet::Tensor<T>& operator[](int64_t index) const {
                // data has to be contiguous to avoid copy operations in the dataloader
                if (index < 0 || index >= data.size() || index >= get_length())
                    throw std::out_of_range("Dataset: Index out of range");
                if (!data[index].is_contiguous())
                    throw std::runtime_error("Dataset: Data is not contiguous");
                return data[index];
            }
        };

    }
}

#endif
