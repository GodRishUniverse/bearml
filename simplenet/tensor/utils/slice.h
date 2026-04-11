#pragma once
#include <vector>

namespace simplenet {
    namespace utils {

        using ull = unsigned long long int;
        class SliceReturn {
            public:
                std::vector<int> shape;
                std::vector<int> strides;
                ull offset;
                SliceReturn(std::vector<int> shape, std::vector<int> strides, ull offset) : shape(shape), strides(strides), offset(offset) {}
        };

        enum class SliceMode {
            FULL,
            RANGE
        };

        class Slice {

            unsigned int _step;
            unsigned int _start;
            unsigned int _end;

            SliceMode _mode;


            public:
                // constructor
                Slice(unsigned int start, unsigned int end, unsigned int step, SliceMode mode = SliceMode::RANGE) : _start(start), _end(end), _step(step), _mode(mode) {}

                // can also be used for setting the step (dual getter/setter)
                unsigned int& get_set_step() { return _step; }
                unsigned int& get_set_start() { return _start; }
                unsigned int& get_set_end() { return _end; }

                // returns a slice that covers the full range (will need to be checked by calling code as only step and Mode will matter)
                static Slice full() {
                    return Slice(0, 0, 1, SliceMode::FULL);
                }
        };
    }
}
