#include "optimizers.h"

// SGD / Adam are class templates; their member definitions live in optimizers.h
// so they are visible at every instantiation point. This translation unit is
// kept only so the existing build target has a source file to compile.
