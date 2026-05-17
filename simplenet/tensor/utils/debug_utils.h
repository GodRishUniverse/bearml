#pragma once
#include <string>
#include <vector>
#include <span>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <cstdlib>

namespace simplenet {
    namespace utils {
        static std::string debugShapes(std::vector<int> shapePassed){
            std::string shape_lit = "";
            for (size_t i = 0; i <shapePassed.size(); i++){
                shape_lit+= std::to_string(shapePassed[i]) + ", ";
            }
            return shape_lit;
        }

        static std::string debugShapes(std::span<int> shapePassed){
            std::string shape_lit = "";
            for (size_t i = 0; i <shapePassed.size(); i++){
                shape_lit+= std::to_string(shapePassed[i]) + ", ";
            }
            return shape_lit;
        }

        template <typename T>
        std::string print_type() {
            // __PRETTY_FUNCTION__ contains the full signature, including the type
            // (kept as a note: that prints the whole function signature, e.g.
            //  "void simplenet::utils::print_type() [with T = double]" — too noisy
            //  to embed in tensor output, and writes to std::cout instead of the
            //  target stream. Instead we return just the demangled element type.)
            int status = 0;
            char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
            std::string name = (status == 0 && demangled) ? demangled : typeid(T).name();
            std::free(demangled);
            return name;
        }

    }
}
