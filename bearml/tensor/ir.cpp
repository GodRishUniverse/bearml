#include "ir.h"
#include <stdexcept>

namespace bearml {

    // one line of IR text for a node - use dtype_name() for the element type
    std::string to_string(const IRNode& n) {
        throw std::logic_error("not implemented: to_string(IRNode)");
    }
}
