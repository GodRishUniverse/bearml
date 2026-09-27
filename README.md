# BearML/bareml: a `mini-pytorch` in pure C++/CUDA

Deep Learning Framework: Implementing Neural Networks in C++

Bare-metal deep learning, hence the bear.

## Why?

I want to implement a project in C++ and CUDA (AMD HIP/ROCm may also be implemented - **optional right now**) properly and this would enable me to do that.

C++ is very fast as compared to Python 3.

Inspired by [llm.c](https://github.com/karpathy/llm.c) from Andrej Karpathy and the [PyTorch](https://github.com/pytorch/pytorch) repository

## Quick Start

### Prerequisites
**Only tested on Fedora Linux at the moment**
- C++20 or higher (**C++23 if you want to use std::bfloat_t**)
- NVCC (NVIDIA CUDA Compiler) uses C++17 mode by default, so C++23 features are not available in CUDA code
- CMake 3.30+
- CUDA Toolkit 13.0 - optional (again will have to make CUDA version optional or support multiple versions)
- GCC 15 (default host compiler) or Clang (pass `-DBEARML_COMPILER=clang`) - CUDA `.cu` files always compile against g++-15 as the nvcc host compiler regardless of this setting, since nvcc's clang host support is version-fragile
- Eigen (included as a third-party submodule, used for CPU BLAS)
- FFTW3 (included as a third-party submodule, used for FFT convolution) - **NOT IMPLEMENTED YET**
- Boost (for string manipulation utilities)
- Google Test (included as a third-party submodule, used for unit tests)
- Tested on Fedora 44 with GCC 15 and CUDA Toolkit 13.0 so far ... **TODO: make OS agnostic**

### Building

```bash
git clone --recursive https://github.com/GodRishUniverse/BearML.git
cd BearML
mkdir build && cd build
cmake ..
make
```

To build with Clang instead of GCC for host code:
```bash
cmake -DBEARML_COMPILER=clang ..
make
```
CUDA `.cu` files still compile through g++-15 as nvcc's host compiler in this mode; only host-side C++/C compilation switches to clang/clang++.

Please edit the CMake files according to your liking.

### Building with Docker/Podman

A `Dockerfile` is provided so you don't need GCC 15/CMake/Boost/etc. installed
locally. It builds a CPU-only image by default (see the commented-out CUDA
block in the `Dockerfile` if you have an NVIDIA GPU, driver, and the NVIDIA
Container Toolkit on the host).

```bash
git clone --recursive https://github.com/GodRishUniverse/BearML.git
cd BearML

# Docker
docker build -t bearml .
docker run --rm -it bearml

# Podman
podman build -t bearml .
podman run --rm -it bearml
```

This drops you into a shell inside the container with the project already
built at `/bearml/build`. From there:
```bash
./build/runchecks/test_runner    # main testing driver
./build/unit_tests/unit_tests    # gtest-based unit tests
```

### Running

After building, the test driver is available at:
```bash
./runchecks/test_runner    # main testing driver (run from the build directory)
```

Unit tests can be run with:
```bash
./unit_tests.sh          # runs gtest-based unit tests
```

## Core Components

### Tensor (`bearml::Tensor<T>`)

The fundamental data structure. `Tensor` is a **class template** parameterized on the
element type. It is multi-dimensional, device-aware (CPU/CUDA), and supports broadcasting.

**dtype aliases** (defined in `tensor/Tensor.h`):

| Alias     | Underlying type  | Description                     |
|-----------|------------------|---------------------------------|
| `TensorD` | `Tensor<double>` | 64-bit float (default / legacy) |
| `Tensorf` | `Tensor<float>`  | 32-bit float                    |
| `TensorBF`| `Tensor<std::bfloat16_t>`  | 16-bit bfloat16       |
| `TensorI` | `Tensor<int>`    | 32-bit signed int               |

Use an alias (e.g. `bearml::Tensorf`) or the explicit form `bearml::Tensor<float>`.
The examples below use `Tensorf`; substitute another alias for a different dtype.

```cpp
#include "bearml.h"

// Create tensors
bearml::Tensorf a({2, 3, 4});         // 2x3x4 tensor, zeros on CPU
bearml::Tensorf b({3, 4});            // 3x4 tensor

// Fill with data
a.linspace(1, 10);                       // fill with linearly spaced values
b.fill(2.0);                             // fill with a constant

// Element-wise operations (broadcasting supported)
auto c = a + b;
auto d = a * b;
auto e = a - b;

// Scalar operations
auto f = 2.0 * a;
auto g = a * 3.5;

// Shape operations
a.reshape({6, 4});
auto t = a.transpose();                  // swap the last two dims (a view)
auto flat = a.flatten(0, -1, true);      // flatten dimensions
a.squeeze(0);                            // remove dimension of size 1

// Reductions
auto s = a.accumulate(1, bearml::reductions::ReductionOps::SUM, false);  // sum along axis 1

// Comparisons
auto mask = bearml::linear_algebra::mask_of_greater_than(a, b, 1.0f, 0.0f);
// any values work, e.g. an attention-style mask:
auto attn = bearml::linear_algebra::mask_of_greater_than(a, b, 0.0f, -std::numeric_limits<float>::infinity());

// Slicing
auto sliced = a.slice("1, 1:5:2");       // numpy-style slicing
auto cont = bearml::Tensorf::contiguous(sliced);

// Concatenation
auto cat = bearml::Tensorf::concat({a, b}, 0);

// Math functions
auto t_out = bearml::Tensorf::tan(a);

// Print
std::cout << a << std::endl;
a.printShape();
bearml::Tensorf::setPrintPrecision(4);  // set decimal precision
```

### Device Management

Tensors can be moved between CPU and CUDA devices.

```cpp
bearml::Device cpu_dev = bearml::Device::cpu();
bearml::Device gpu_dev = bearml::Device::cuda(0);  // GPU 0

bearml::Tensorf t({3, 4});
t.to_(gpu_dev);              // move to GPU in-place
auto t_cpu = t.to(cpu_dev);  // create a copy on CPU
```

### Autograd

Reverse-mode automatic differentiation is built into `Tensor`. Mark a tensor with
`set_requires_grad()`; every op on it records a graph node (op code + attributes) on the
result, and `backward` walks that graph with a gradient table (`grad_of`). Tensors that are
not tracked cost nothing extra. Autograd works on floating tensors (`Tensorf`, `TensorD`,
`TensorBF`); scalars are 1-element tensors.

```cpp
// Scalar autodiff - a scalar is a 1-element tensor
bearml::TensorD x({1}); x.fill(4.0); x.set_requires_grad();
bearml::TensorD y({1}); y.fill(2.0); y.set_requires_grad();
auto z = bearml::linear_algebra::hadamard(x, y) + x;     // z = x*y + x
// dz/dx = y + 1 = 3, dz/dy = x = 4

bearml::autogradient::backward(z);
std::cout << x.grad() << std::endl;  // 3.0
std::cout << y.grad() << std::endl;  // 4.0

// Tensor autodiff
bearml::Tensorf a({2, 3});
a.linspace(1, 6);
a.set_requires_grad();
auto result = bearml::linear_algebra::hadamard(a, a);  // element-wise square
bearml::autogradient::backward(result);
// a.grad() now contains 2*a
```

| API | What it does |
|-----|--------------|
| `t.set_requires_grad(bool = true)` | make `t` a trainable leaf (or stop tracking it) |
| `t.requires_grad()` / `t.grad()` / `t.zero_grad()` | query, read (throws if untracked) and reset the gradient |
| `autogradient::backward(root, accumulate = false)` | backprop from `root`; `accumulate = true` adds onto the leaves' existing grads |
| `autogradient::gradients(root, {&w, &b})` | backprop and return the gradients of the listed tensors |
| `t.detach()` | same data, no graph - gradients stop here |
| `t.shared_view()` | same data and same graph node |

Notes: `*` on two tensors is matrix multiplication - use `linear_algebra::hadamard` for
element-wise products. Copies are deep and untracked; moves keep the node. In-place ops
(`+=`, `fill`, `to_`, ...) are not recorded and throw on tracked intermediate results.
Backward is not itself recorded, so higher-order derivatives are not supported yet.

### Neural Network Modules

PyTorch-style module system with polymorphism.

**Base class:** `bearml::neural_network::Module<T>` (abstract, `T` defaults to `Tensorf`)
- `T forward(T& x)` - forward pass (pure virtual); `operator()` calls it
- `std::vector<std::shared_ptr<T>> parameters()` - the layer's trainable tensors, shared with the optimizer
- Xavier and He initialization built in
- the input must be on the layer's device, otherwise `forward` throws

**Available layers:**
- `Linear(in_features, out_features, init_method, device, seed)` - fully connected layer
- `ReLU()` - rectified linear unit
- `Sigmoid()` - sigmoid activation
- `LeakyReLU(negative_slope)` - leaky ReLU
- `Tanh()` - hyperbolic tangent
- `Softmax(dim)`, `GELU()`, `SiLU()` / `Swish`, `SoftPlus()`

Copying a `Linear` gives an independent layer (its own weights and gradients); moving it
keeps the weights, so an optimizer built on it stays valid.

```cpp
bearml::Device dev = bearml::Device::cuda(0);

// Create layers - all on the same device as the input
bearml::neural_network::Linear<> fc1(784, 128, "Xavier", dev);
bearml::neural_network::ReLU<> relu(42, dev);
bearml::neural_network::Linear<> fc2(128, 10, "He", dev);

// Forward pass through layers
auto h = fc1(input);            // input: Tensorf of shape {batch, 784} on dev
auto h_act = relu(h);
auto out = fc2(h_act);
```

### Model Construction (`Model_Construct`)

Base class for defining custom models (analogous to `torch.nn.Module`).

```cpp
class MyModel : public bearml::neural_network::Model_Construct {
public:
    bearml::neural_network::Linear<> layer1;
    bearml::neural_network::Tanh<> activation;
    bearml::neural_network::Linear<> layer2;

    MyModel(int in_size, int out_size, bearml::Device dev = bearml::Device::cpu())
        : layer1(in_size, 64, "Xavier", dev),
          activation(42, dev),
          layer2(64, out_size, "Xavier", dev) {}

    bearml::Tensorf forward(std::vector<bearml::Tensorf> inputs) override {
        auto& x = inputs[0];
        auto h = layer1(x);
        auto h_act = activation(h);
        return layer2(h_act);
    }

    std::vector<std::shared_ptr<bearml::Tensorf>> parameters() override {
        auto params = layer1.parameters();
        auto l2_params = layer2.parameters();
        params.insert(params.end(), l2_params.begin(), l2_params.end());
        return params;
    }
};
```

### Loss Functions

Located in `bearml::neural_network::loss_functions`:

| Function | Description |
|----------|-------------|
| `l1_loss(actual, predictions)` | Mean Absolute Error (MAE) |
| `l2_loss(actual, predictions)` | Mean Squared Error (MSE) |
| `log_loss(actual, predictions)` | Binary Cross-Entropy (Log Loss) |
| `bce_loss_with_logits(actual, predictions)` | currently the same as `log_loss` |
| `cross_entropy_loss(actual, predictions)` | softmax over the last dim, then cross entropy |

Losses take tensors and return a tracked tensor; `actual` is usually untracked.

```cpp
auto loss = bearml::neural_network::loss_functions::l1_loss(actual, predictions);
bearml::autogradient::backward(loss);
```

### Optimizers

Located in `bearml::neural_network::optimizers`:

| Optimizer | Parameters |
|-----------|------------|
| `SGD(params, lr)` | Learning rate (default: 0.0001) |
| `Adam(params, lr, beta1, beta2, eps)` | LR, momentum decay, RMSProp decay, epsilon |

```cpp
bearml::neural_network::optimizers::SGD optim(model.parameters(), 0.01);
// or
bearml::neural_network::optimizers::Adam optim(model.parameters(), 0.001);

for (int epoch = 0; epoch < 100; epoch++) {
    optim.zero_grad();

    auto pred = model.forward({input});
    auto loss = bearml::neural_network::loss_functions::l2_loss(target, pred);

    bearml::autogradient::backward(loss);
    optim.step();

    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
}
```

### CUDA Kernels

The CUDA backend provides templated GPU kernels for all supported types:
- **Floating point:** `float`, `double`, `__half` (fp16), `__nv_bfloat16` (bf16)
- **Integer:** `int8_t`, `int16_t`, `int32_t`, `int64_t`

**Available kernels:**
- Element-wise operations (add, sub, mul, div, pow, etc.) with broadcasting support
- Matrix multiplication (naive GEMM)
- Reductions (sum, product via atomicCAS-based `atomicMul`)
- Transpose
- Fill
- Padding (constant, reflect, replicate)
- Comparison operations
- Equality checks

Kernels are dispatched automatically when tensors are on a CUDA device. The host-side dispatch logic lives in `cuda_kernels.h/.cpp` and the kernel implementations are in the `kernels/` directory.

---

## End-to-End Training Example

```cpp
#include "bearml.h"
#include <iostream>

class Model : public bearml::neural_network::Model_Construct {
public:
    bearml::neural_network::Linear<> layer1;
    bearml::neural_network::Tanh<> nonlinearity;
    bearml::neural_network::Linear<> layer2;

    Model(int in_shape, int out_shape, bearml::Device dev = bearml::Device::cpu())
        : layer1(in_shape, out_shape, "Xavier", dev),
          nonlinearity(42, dev),
          layer2(out_shape, out_shape, "Xavier", dev) {}

    bearml::Tensorf forward(std::vector<bearml::Tensorf> inputs) override {
        auto& x = inputs[0];
        auto f1 = layer1(x);
        auto f2 = nonlinearity(f1);
        return layer2(f2);
    }

    std::vector<std::shared_ptr<bearml::Tensorf>> parameters() override {
        std::vector<std::shared_ptr<bearml::Tensorf>> params;
        auto l1 = layer1.parameters();
        params.insert(params.end(), l1.begin(), l1.end());
        auto l2 = layer2.parameters();
        params.insert(params.end(), l2.begin(), l2.end());
        return params;
    }
};

int main() {
    bearml::Device dev = bearml::Device::cuda(0);

    Model model(2, 5, dev);

    bearml::Tensorf input({1, 2});
    input.linspace(1, 2);
    input.to_(dev);

    bearml::Tensorf target({1, 5});
    target.linspace(1, 5);
    target.to_(dev);

    bearml::neural_network::optimizers::SGD optim(model.parameters(), 0.1);

    for (int i = 0; i < 10; i++) {
        optim.zero_grad();

        auto pred = model.forward({input});
        auto loss = bearml::neural_network::loss_functions::l1_loss(target, pred);

        bearml::autogradient::backward(loss);
        optim.step();

        std::cout << "Loss: " << loss << std::endl;
    }

    return 0;
}
```

---

## What has been done

* Multi-dimensional Tensor class with broadcasting, slicing, and device awareness
* Element-wise operations: add, subtract, multiply, divide, pow, exp, log, sign, abs, trig functions
* Batched matrix multiplication (GEMM) on CPU (via Eigen) and GPU (naive CUDA kernel)
* Multi-dimensional transpose
* Tensor reductions: sum, product (with CUDA atomicCAS-based atomicMul)
* Kahan summation for numerical stability
* Templated `Tensor<T>` class with dtype aliases (`TensorD`, `Tensorf`, `TensorBF`, `TensorI`)
* Reverse-mode automatic differentiation built into `Tensor` (graph nodes as op code + attributes, one gradient table)
* Template bodies in `.cpp` files with explicit instantiation - editing an op rebuilds one file
* Neural network module system: Linear, ReLU, Sigmoid, LeakyReLU, Tanh, Softmax, GELU, SiLU, SoftPlus
* Model construction base class (`Model_Construct`)
* Loss functions: L1 (MAE), L2 (MSE), Log Loss (BCE)
* Optimizers: SGD, Adam (AdamW)
* Xavier and He weight initialization
* CUDA backend with templated kernels for all data types (fp16, bf16, float, double, int8-int64)
* Tensor padding (constant, reflect, replicate)
* Comparison masks (greater than, less than, equal, etc.)
* Numpy-style tensor slicing
* Tensor concatenation
* CPU/GPU memory management with device transfer (`to_`, `to`)
* Can train a basic neural network on both CPU and CUDA

---

## What we need to do

### CUDA Support & Core Infrastructure
* **Refactoring for CUDA Support (Major Overhaul)** -- **IN PROGRESS (Priority)**
    * Write host code and kernels -- **Status: Kernels in progress**
      * kernels need to be fixed for when the datasize is larger than the number of threads (this wont compute it) AS THREAD_IDX WILL NEVER REACH N
    * Call kernels in the `Tensor` class when devices match (CUDA). **DOING alongside CUDA support**
    * Implement **Lazy Copy** operation.
    * **Memory Management:** Address CPU/GPU memory usage. (Decision needed: Single memory space vs. syncing to avoid expensive copy operations).
* **Refactoring for Slice Support** - thinking about it
    * All ops will have to become slice aware and a `::contiguous` static function will be needed
    * This will allow easy implementation of convolution operations (i think) - but also nice way to have slicing support

* **Dependency Management:** Fix `#includes` for the repo to remove cyclical dependencies and repeated includes. -- **Priority After Kernels**
* **Templatify**
  * ~~**Templatify Tensor:** template the `Tensor` class to support different data types.~~ **DONE** (`Tensor<T>` with `TensorD`/`Tensorf`/`TensorI` aliases)
  * ~~**Templatify Autodiff:** template the autodiff engine for different data types.~~ **DONE** (generic over `Tensor<T>` via `is_tensor_v`)
  * ~~**Templatify Kernel:** template the CUDA kernels for different data types.~~ **DONE** (kernels templated, instantiated per element type)
  * Explicit-instantiation `.cu` split + `if constexpr` host-compute guards so non-`double` aliases can be instantiated -- **IN PROGRESS**
* Extend supported dtypes (`int8, int16, int64, float16, bfloat16`, etc.) and allow Mixed Precision
* Add OpenMP support for CPU side
* **Hardware Support:** Potential support for AMD HIP/ROCm. -- **Maybe**

### Math Engine & Tensor Operations
* **Matrix Multiplication (GEMM):** Use [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) integration.
    * Device-aware execution: Check device string to call CUDA kernel or standard Matmul.
    * Use CUDA for GEMM and Matmul. **Naive kernel is made**
* **Vector Operations:** Rectify `Transpose` for vector operations (column vs. row).
    * Apply corresponding modifications to multiplication in `autogradient.cpp`.
* **Caching:** Implement tensor caching to reduce memory usage.
* **Convolution:** Implement tensor convolution support.

### Autogradient/Autodiff & Neural Network Module
* **Autodiff Engine:** Implement for activation functions (Unary and Binary ops).
* **Sequential Class:** Create a wrapper to stack layers.
* **Loss Functions:**
    * Cross Entropy Loss (using LogSoftmax - fix needed)
    * Softmax Loss
    * Cross Entropy Loss with Softmax
    * Binary Cross Entropy Loss with Sigmoid
* **Optimizers (Post-Loss Implementation):**
    * **SGD:** Currently implemented but slow. Add Momentum.
    * **Adam / AdamW:** Implement Eps, Betas, and Regularization.

### Data & Pipeline
* **Dataloading Pipeline:** Support for shuffling and batching.
    * Modular design for different types (Images, CSV, etc.).
* **Model Persistence:** Saving and loading pipelines.
  * Safetensor file loading as well

### Testing
* Unit Tests using `gtest` (framework integrated, tests in progress)

### Optimizations
* **CPU Optimizations:**
    * Vectorization using AVX/AVX2 intrinsics (e.g., `axpy` for SGD).
    * Matrix Multiplication Tiling (Cache blocking).
    * OpenMP parallelization (specifically for Matmul).
* **Graph & Performance:**
    * Lazy evaluation of the computational graph.
    * Operator Fusing (e.g., ReLU + Linear).
    * Strassen's algorithm for large matrices.
    * Compile-time graph optimization (e.g., `torch.compile` equivalent using **XLA and MLIR**) - **Need to research**
* **Advanced Features:**
    * Mixed precision training -- **High Importance**
    * Dropout.
    * He initialization (to supplement current Xavier default).
* **Cleanup:** Removing Eigen operations **(Long-term)**.

### Compatibility and Containerization
* ~~Provide a Dockerfile for building the project.~~ Done (CPU-only by default, see `Dockerfile`) - still needs a proper CUDA-enabled base image/variant.
* ~~Add compatibility for `clang` compiler.~~ Done (`-DBEARML_COMPILER=clang`, host code only; CUDA still built via g++-15). MSVC still to do.
* Add OS agnostic support (e.g., cross-platform compatibility).

### EXPLAINATION
* Create a website that explains the project and its features.
* Update readme with explanations of optimizations and features and diagrams

---

## Contributing

This repository is open to contributions. Please make an issue before submitting a pull request.

### Getting Started

1. Fork and clone the repository (with submodules):
   ```bash
   git clone --recursive https://github.com/GodRishUniverse/BearML.git
   cd BearML
   ```

2. Create a build directory and build:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. Run the test driver to verify your build:
   ```bash
   ./runchecks/test_runner
   ```

4. Run unit tests:
   ```bash
   cd .. && ./unit_tests.sh
   ```

### Making Changes

**Adding a new CUDA kernel:**

1. Create `your_kernel.cu` and `your_kernel.cuh` in `bearml/cuda/kernels/`.
2. In the `.cuh` file, declare the kernel function and the `launch_*` host wrapper. Use the existing pattern with `INSTANTIATE_*` macros for template instantiation.
3. In the `.cu` file, implement the `__global__` kernel and the `launch_*` function. Follow the existing pattern for stream management (create a stream if `nullptr` is passed, synchronize and destroy if owned).
4. Add forward declarations in `bearml/cuda/includes/kernel_links.h` so host code can call your launcher.
5. Add the `.cu` file to `bearml/CMakeLists.txt` in the source list.

**Adding a new layer/activation:**

Template bodies live in `.cpp` files and are explicitly instantiated at the bottom of each
file (`Tensor`/`linalg_utils`: `float`, `double`, `int`, `bfloat16`; layers, losses and
optimizers: `Tensorf`, `TensorD`). Headers keep declarations plus `extern template` lines.
When you add a class or function template, add its instantiation lines too, or you get
link errors.

1. Create your class inheriting from `bearml::neural_network::Module<T>`: declare it in a header in `bearml/activation_functions/`, put the member bodies in the matching `.cpp`.
2. Implement `T forward(T& x)` (call `this->check_device(x)` first), `T operator()(T& x)` and `std::vector<std::shared_ptr<T>> parameters()`.
3. Add `template class ...<bearml::Tensorf>;` / `<bearml::TensorD>` in the `.cpp` and matching `extern template` lines in the header.
4. Include it in `bearml/bearml.h` if it should be part of the public API.

**Adding a new loss function:**

1. Declare it in `bearml/loss_functions/loss.h` and define it in `loss.cpp`, following the existing pattern, with instantiations for `Tensorf` and `TensorD`.
2. It takes and returns tensors (`T l1_loss(T& actual, T& predictions)`). Build it from tensor ops (`+`, `-`, `/`, `hadamard`, `T::mean`, ...) so gradients flow through automatically.

**Adding a new optimizer:**

1. Inherit from `bearml::neural_network::optimizers::Optimizer` in `bearml/optimizers/optimizers.h`; hold the parameters as `std::vector<std::shared_ptr<T>>`.
2. Implement `step()` and `zero_grad()`.
3. Add the implementation and its instantiations in `optimizers.cpp`.

**Adding new Tensor operations:**

1. If it's a host-side utility, add it to the appropriate file under `bearml/tensor/utils/`.
2. If it needs a CUDA kernel, follow the kernel instructions above and add the host dispatch in `cuda_kernels.h/.cpp`.
3. Declare the method in `Tensor.h` and define it in `Tensor.cpp`.
4. To make it differentiable: call `out.record_op(OP_Code::..., {&inputs...}, attrs, out)` after computing, add an `OP_Code` in `operators/ops.h`, and add its gradient rule to `grad_of` in `autograd/autogradient.cpp`.

### Build Configuration

The project uses CMake with the following defaults (see top-level `CMakeLists.txt`):
- C++ standard: C++23 (host), C++17 (CUDA)
- Host compiler: GCC 15, found via `find_program(NAMES gcc-15 gcc)` / `g++-15 g++`
- CUDA Toolkit: 13.0

Pass `-DBEARML_COMPILER=clang` to build host code with Clang/Clang++ instead
(`find_program(NAMES clang)` / `clang++`). CUDA `.cu` files always compile
against g++-15 as the nvcc host compiler either way. Set `-DBEARML_USE_CUDA=OFF`
for a CPU-only build.

You may need to adjust the compiler paths and CUDA toolkit version for your system.

### NOTE and manual patch-work

CUDA compilation (manual patch applied)
`/usr/local/cuda-13/targets/x86_64-linux/include/crt`
as suggested by "https://www.linuxquestions.org/questions/slackware-14/help-compiling-ffmpeg-8-with-cuda-12-9-using-an-alternative-glibc-4175754496-print/"
changed math libraries to
```c
#if defined(__GLIBC__) && (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 42)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double rsqrt(double x) noexcept (true);
#else
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double rsqrt(double x);
#endif

#if defined(__GLIBC__) && (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 42)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float rsqrtf(float x) noexcept (true);
#else
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float rsqrtf(float x);
#endif
```

`__restrict__` keyword usage: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/

---

## Author

Rishabh Agarwal ([@GodRishUniverse](https://github.com/GodRishUniverse))

## Citations [will formalize]

>
> [1] [Thank you u/brandonpelfrey](https://www.reddit.com/r/algorithms/comments/1naehk1/comment/ndpkcqr/)
>
> [2] [max() derivative formula](https://math.stackexchange.com/questions/368432/derivative-of-max-function)
>
> [3] [convolution derivative formula](https://math.stackexchange.com/questions/177239/derivative-of-convolution)
>
> [4] [intuitive convolution](https://betterexplained.com/articles/intuitive-convolution/)
>

## AI Disclosure

Claude code was used for refactoring and understanding concepts and add comments in some places (I highly encourage you to read the code!!!). I wrote most of my code myself and you can check Git history for that.
