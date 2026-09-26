#pragma once
#include <stdint.h>
#include <stddef.h>
#include "padding_ops.h"

#ifndef OP_CODE
#define OP_CODE

struct OpAttributes{
    int dim = -1;                                      // softmax
    int pad_amount = 0;
    Padding_Op_Code pad_mode = Padding_Op_Code::PAD_CONSTANT;
    double constant = 0;                               // scalar operand
    bool constant_on_left = false;                     // 2.0 / x vs x / 2.0
};


enum class OP_ARITY : uint16_t {
    NONE = 0,
    UNARY = 1,
    BINARY = 2,
};

// operation (op) codes for the operations that will be done element-wise
enum class OP_Code : uint16_t {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MAX,
    OP_MIN,
    OP_HADAMARD,
    // unary operations
    OP_EXP,
    OP_TRANSPOSE,
    OP_LOG,
    OP_ABS,
    OP_SQRT, // TODO: Implement
    OP_MEAN_FOR_GRAD,
    OP_SOFTMAX,
    // trigonometric operations (unary)
    OP_SIN,
    OP_COS,
    OP_TAN,
    // hyperbolic operations (unary)
    OP_SINH,
    OP_COSH,
    OP_TANH,

    // ARB
    OP_PAD,
    OP_SUM,

    // scalar operand: one tensor input, constant in OpAttributes
    OP_ADD_SCALAR,                                     // x + c, c + x
    OP_SUB_SCALAR,                                     // x - c
    OP_RSUB_SCALAR,                                    // c - x
    OP_MUL_SCALAR,                                     // x * c, c * x
    OP_DIV_SCALAR,                                     // x / c
    OP_RDIV_SCALAR,                                    // c / x

    // OP NULL
    NO_OP // No operation
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
