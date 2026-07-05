#pragma once
#include <cstdint>

namespace bearml {
    enum class CompareOp: uint16_t {
        GT,
        GE,
        LT,
        LE,
        EQ,
        NE,
        SIGN_OP
    };
}
