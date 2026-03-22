#pragma once
#include <cstdint>

namespace simplenet {
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
