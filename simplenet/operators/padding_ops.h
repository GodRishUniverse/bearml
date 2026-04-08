#pragma once
#include <stdint.h>
#include <stddef.h>


#ifndef PADDING_OP_CODE
#define PADDING_OP_CODE

// operation (op) codes for the operations that will be done element-wise
enum class Padding_Op_Code : uint16_t {
    PAD_ZERO
};
#endif
