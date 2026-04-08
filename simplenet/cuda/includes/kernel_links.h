// Never add ".cu" files here as GCC doesnt recognize them
// It recognizes ".cuh" as CUDA header files so the NVCC can compile the ".cu" files

#include "../kernels/element_wise_kernels.cuh"
#include "../kernels/fill.cuh"
#include "../kernels/matmul.cuh"
#include "../kernels/sum.cuh"
#include "../kernels/checks.cuh"
#include "../kernels/transpose.cuh"
#include "../kernels/compares.cuh"
#include "../kernels/padding.cuh"
