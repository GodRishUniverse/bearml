#pragma once
#include <string>
#include <vector>

namespace simplenet {
    namespace utils {
        static std::string debugShapes(std::vector<int> shapePassed){
            std::string shape_lit = "";
            for (size_t i = 0; i <shapePassed.size(); i++){
                shape_lit+= std::to_string(shapePassed[i]) + ", ";
            }
            return shape_lit;
        }

    }
}
