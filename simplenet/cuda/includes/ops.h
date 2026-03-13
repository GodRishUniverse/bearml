#pragma once
#include <stdint.h>
#include <stddef.h>


#ifndef CUDA_OP_CODE
#define CUDA_OP_CODE

// operation (op) codes for the operations that will be done element-wise
enum class OP_Code : uint16_t {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_EXP,
    OP_LOG,
};
#endif
