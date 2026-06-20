#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "tensor/Tensor.h"


#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

// read csv files

namespace simplenet {
    namespace data {

        inline simplenet::TensorD read_csv(const std::string& file_path, bool has_header = true){
            std::vector<double> row; int cols = 0;
            std::ifstream f(file_path); std::string line;
            if (!f) throw std::runtime_error("read_csv: cannot open the file " + file_path);
            if (has_header) std::getline(f, line); // discard header row
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                std::stringstream ss(line); std::string cell; int c = 0;
                while (std::getline(ss, cell, ',')) { row.push_back(std::stod(cell)); ++c; }
                cols = c;
            }
            if (cols == 0) throw std::runtime_error("read_csv: no data parsed from " + file_path);
            int rows = static_cast<int>(row.size()) / cols;
            return simplenet::TensorD::from_host({rows, cols}, row);
        }
    }
}

#endif
