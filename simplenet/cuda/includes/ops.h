#pragma once
#include <stdint.h>
#include <stddef.h>

#ifndef CUDA_OP_CODE
#define CUDA_OP_CODE

// operation (op) codes for the operations that will be done element-wise
enum OP_Code : uint16_t {
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3,
};
#endif
