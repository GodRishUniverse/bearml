#pragma once
#include <stdint.h>
#include <stddef.h>


#ifndef OP_CODE
#define OP_CODE

// operation (op) codes for the operations that will be done element-wise
enum class OP_Code : uint16_t {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MAX,
    OP_MIN,
    // unary operations
    OP_EXP,
    OP_LOG,
    OP_ABS,
    OP_SQRT,
    // trigonometric operations (unary)
    OP_SIN,
    OP_COS,
    OP_TAN,
    // hyperbolic operations (unary)
    OP_SINH,
    OP_COSH,
    OP_TANH
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
