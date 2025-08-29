#include "shape_utils.h"

namespace simplenet {
    namespace utils {
        bool negOrZeroInSizeCheck(const std::vector<int>& sizePassedDown) {
            for (const int & i : sizePassedDown){
                if (i < 1){
                    return true;
                }
            }
            return false;
        };
    }
}
