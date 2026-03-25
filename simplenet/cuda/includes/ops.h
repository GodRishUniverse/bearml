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
    OP_MAX,
    OP_MIN,
    OP_ABS,
    OP_SQRT
};
#endif


#ifndef LHS_RHS_CODE
#define LHS_RHS_CODE

// operation (op) codes for the operations that will be done element-wise
enum class LHS_RHS_Code : uint16_t {
    OP_RHS,
    OP_LHS
};
#endif
