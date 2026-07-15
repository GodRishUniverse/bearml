// TODO: look into XLA and MLIR and how high level ops (hlo) are represented
// will be used by StaticTensor class
#pragma once

// Need to look over : https://github.com/openxla/stablehlo
// will use stablehlo - will need to add check if compiler is clang as I need LLVM backend for stablehlo and XLA

// I think I will have to defer to not using this IR right now as I need to implement and understand how lazy tensor materialization works
// I can use MLIR;s toy tutorial to learn it better before implementing it
