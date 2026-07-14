// We will create a tensor class (will see if we inherit from Tensor class or not) to be compiled
// Help with graph keeping and operator fusing of ops
// Can use XLA for faster execution as everything is precompiled
// TODO: need to do more research into XLA/MLIR and ML compilers and how they can be used to compile tensor ops


#pragma once
#include "Tensor.h"

// TODO: need to think about how to implement this and design this so that we can use it correctly especially - it should mirror Tensor EXACTLY
namespace bearml {
    // TODO: check how to implement this and design this

    template <typename T>
    class StaticTensor : public Tensor<T> {
        public:
            // we need incoming op and output op to know how to fuse

    };
}
