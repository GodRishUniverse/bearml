#pragma once
#include <vector>

namespace simplenet {
    namespace utils {

        using ull = unsigned long long int;
        class SliceReturn {
            public:
                std::vector<int> shape;
                int offset;
                SliceReturn(std::vector<int> shape, int offset) : shape(shape), offset(offset) {}
        };

        enum class SliceMode {
            FULL,
            RANGE
        };

        class Slice {

            int _step;
            int _start;
            int _end;

            SliceMode _mode;


            public:
                // constructor
                Slice(int start, int end, int step, SliceMode mode = SliceMode::RANGE) : _start(start), _end(end), _step(step), _mode(mode) {}

                // can also be used for setting the step (dual getter/setter)
                int& get_set_step() { return _step; }
                int& get_set_start() { return _start; }
                int& get_set_end() { return _end; }

                // returns a slice that covers the full range (will need to be checked by calling code as only step and Mode will matter)
                static Slice full() {
                    return Slice(0, 0, 1, SliceMode::FULL);
                }
        };

        // function declaration
        SliceReturn computing_slice_parameters(std::vector<int> shape, std::vector<int> strides, std::vector<utils::Slice> slices);
    }
}
