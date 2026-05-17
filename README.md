# SimpleNet: a `mini-pytorch` in pure C++/CUDA

Deep Learning Framework: Implementing Neural Networks in C++

## Why?

I want to implement a project in C++ and CUDA (AMD HIP/ROCm may also be implemented - **optional right now**) properly and this would enable me to do that.

C++ is very fast as compared to Python 3.

Inspired by [llm.c](https://github.com/karpathy/llm.c) from Andrej Karpathy and the [PyTorch](https://github.com/pytorch/pytorch) repository

## Quick Start

### Prerequisites
**Only tested on Fedora Linux at the moment**
- C++20 or higher (check NVCC compatibility)
- CMake 3.30+
- CUDA Toolkit 13.0 (will have to make optional)
- GCC 14 (configured as the default compiler)
- Eigen (included as a third-party submodule, used for CPU BLAS)
- Boost (for string manipulation utilities)
- Google Test (included as a third-party submodule, used for unit tests)

### Building

```bash
git clone --recursive https://github.com/GodRishUniverse/SimpleNet.git
cd SimpleNet
mkdir build && cd build
cmake ..
make
```

Please edit the CMake files according to your liking.

### Running

After building, the test driver is available at:
```bash
./runchecks/runchecks    # main testing driver
```

Unit tests can be run with:
```bash
./unit_tests.sh          # runs gtest-based unit tests
```

## Core Components

### Tensor (`simplenet::Tensor<T>`)

The fundamental data structure. `Tensor` is a **class template** parameterized on the
element type. It is multi-dimensional, device-aware (CPU/CUDA), and supports broadcasting.

**dtype aliases** (defined in `tensor/Tensor.h`):

| Alias     | Underlying type  | Description                     |
|-----------|------------------|---------------------------------|
| `TensorD` | `Tensor<double>` | 64-bit float (default / legacy) |
| `Tensorf` | `Tensor<float>`  | 32-bit float                    |
| `TensorI` | `Tensor<int>`    | 32-bit signed int               |

Use an alias (e.g. `simplenet::Tensorf`) or the explicit form `simplenet::Tensor<float>`.
The examples below use `Tensorf`; substitute another alias for a different dtype.

```cpp
#include "simplenet.h"

// Create tensors
simplenet::Tensorf a({2, 3, 4});         // 2x3x4 tensor, zeros on CPU
simplenet::Tensorf b({3, 4});            // 3x4 tensor

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
auto t = simplenet::Tensorf::transpose(a, 0, 1);
auto flat = a.flatten(0, -1, true);      // flatten dimensions
a.squeeze(0);                            // remove dimension of size 1

// Reductions
auto s = a.sum(1, false);               // sum along axis 1

// Comparisons
auto mask = simplenet::linear_algebra::mask_of_greater_than(a, b, 1.0, 0.0);

// Slicing
auto sliced = a.slice("1, 1:5:2");       // numpy-style slicing
auto cont = simplenet::Tensorf::contiguous(sliced);

// Concatenation
auto cat = simplenet::Tensorf::concat({a, b}, 0);

// Math functions
auto t_out = simplenet::Tensorf::tan(a);

// Print
std::cout << a << std::endl;
a.printShape();
simplenet::Tensorf::setPrintPrecision(4);  // set decimal precision
```

### Device Management

Tensors can be moved between CPU and CUDA devices.

```cpp
simplenet::Device cpu_dev = simplenet::Device::cpu();
simplenet::Device gpu_dev = simplenet::Device::cuda(0);  // GPU 0

simplenet::Tensorf t({3, 4});
t.to_(gpu_dev);              // move to GPU in-place
auto t_cpu = t.to(cpu_dev);  // create a copy on CPU
```

### Autograd (`simplenet::Node<T>`)

Reverse-mode automatic differentiation via a computational graph. `Node<T>` is generic
over its value type: it works on scalar `double` and on any `Tensor<U>` (e.g. `Tensorf`,
`TensorD`) — the autograd internals dispatch on the `is_tensor_v` trait rather than a
hardcoded tensor dtype.

```cpp
// Scalar autodiff
auto x = simplenet::Node<double>::make_node(4.0);
auto y = simplenet::Node<double>::make_node(2.0);
auto z = x * y + x;     // z = x*y + x
// dz/dx = y + 1 = 3, dz/dy = x = 4

simplenet::autogradient::backward(z);
std::cout << x->grad << std::endl;  // 3.0
std::cout << y->grad << std::endl;  // 4.0

// Tensor autodiff
simplenet::Tensorf a({2, 3});
a.linspace(1, 6);
auto node_a = simplenet::Node<simplenet::Tensorf>::make_node(a);
auto result = node_a * node_a;  // element-wise square
simplenet::autogradient::backward(result);
// node_a->grad now contains 2*a
```

### Neural Network Modules

PyTorch-style module system with polymorphism.

**Base class:** `simplenet::neural_network::Module` (abstract)
- `forward()` - forward pass (pure virtual)
- `parameters()` - returns trainable parameters
- Xavier and He initialization built in

**Available layers:**
- `Linear(in_features, out_features, init_method, device, seed)` - fully connected layer
- `ReLU()` - rectified linear unit
- `Sigmoid()` - sigmoid activation
- `LeakyReLU(negative_slope)` - leaky ReLU
- `Tanh()` - hyperbolic tangent

```cpp
simplenet::Device dev = simplenet::Device::cuda(0);

// Create layers
simplenet::neural_network::Linear fc1(784, 128, "Xavier", dev);
simplenet::neural_network::ReLU relu;
simplenet::neural_network::Linear fc2(128, 10, "He", dev);

// Forward pass through layers
auto x = simplenet::Node<simplenet::Tensorf>::make_node(input);
auto h = fc1(x);
auto h_act = relu(h);
auto out = fc2(h_act);
```

### Model Construction (`Model_Construct`)

Base class for defining custom models (analogous to `torch.nn.Module`).

```cpp
class MyModel : public simplenet::neural_network::Model_Construct {
public:
    simplenet::neural_network::Linear layer1;
    simplenet::neural_network::Tanh activation;
    simplenet::neural_network::Linear layer2;

    MyModel(int in_size, int out_size, simplenet::Device dev = simplenet::Device::cpu())
        : layer1(in_size, 64, "Xavier", dev),
          activation(42, dev),
          layer2(64, out_size, "Xavier", dev) {}

    std::shared_ptr<simplenet::Node<simplenet::Tensorf>> forward(
            std::vector<simplenet::Tensorf> inputs) override {
        auto x = simplenet::Node<simplenet::Tensorf>::make_node(inputs[0]);
        auto h = layer1(x);
        auto h_act = activation(h);
        return layer2(h_act);
    }

    std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensorf>>> parameters() override {
        auto params = layer1.parameters();
        auto l2_params = layer2.parameters();
        params.insert(params.end(), l2_params.begin(), l2_params.end());
        return params;
    }
};
```

### Loss Functions

Located in `simplenet::neural_network::loss_functions`:

| Function | Description |
|----------|-------------|
| `l1_loss(actual, predictions)` | Mean Absolute Error (MAE) |
| `l2_loss(actual, predictions)` | Mean Squared Error (MSE) |
| `log_loss(actual, predictions)` | Binary Cross-Entropy (Log Loss) |

```cpp
auto actual_node = simplenet::Node<simplenet::Tensorf>::make_node(actual);
auto loss = simplenet::neural_network::loss_functions::l1_loss(actual_node, predictions);
```

### Optimizers

Located in `simplenet::neural_network::optimizers`:

| Optimizer | Parameters |
|-----------|------------|
| `SGD(params, lr)` | Learning rate (default: 0.0001) |
| `Adam(params, lr, beta1, beta2, eps)` | LR, momentum decay, RMSProp decay, epsilon |

```cpp
simplenet::neural_network::optimizers::SGD optim(model.parameters(), 0.01);
// or
simplenet::neural_network::optimizers::Adam optim(model.parameters(), 0.001);

for (int epoch = 0; epoch < 100; epoch++) {
    optim.zero_grad();

    auto pred = model.forward({input});
    auto actual_node = simplenet::Node<simplenet::Tensorf>::make_node(target);
    auto loss = simplenet::neural_network::loss_functions::l2_loss(actual_node, pred);

    simplenet::autogradient::backward(loss);
    optim.step();

    std::cout << "Epoch " << epoch << " Loss: " << loss->val << std::endl;
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
#include "simplenet.h"
#include <iostream>

class Model : public simplenet::neural_network::Model_Construct {
public:
    simplenet::neural_network::Linear layer1;
    simplenet::neural_network::Tanh nonlinearity;
    simplenet::neural_network::Linear layer2;

    Model(int in_shape, int out_shape, simplenet::Device dev = simplenet::Device::cpu())
        : layer1(in_shape, out_shape, "Xavier", dev),
          nonlinearity(42, dev),
          layer2(out_shape, out_shape, "Xavier", dev) {}

    std::shared_ptr<simplenet::Node<simplenet::Tensorf>> forward(
            std::vector<simplenet::Tensorf> inputs) override {
        auto x = simplenet::Node<simplenet::Tensorf>::make_node(inputs[0]);
        auto f1 = layer1(x);
        auto f2 = nonlinearity(f1);
        return layer2(f2);
    }

    std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensorf>>> parameters() override {
        std::vector<std::shared_ptr<simplenet::Node<simplenet::Tensorf>>> params;
        auto l1 = layer1.parameters();
        params.insert(params.end(), l1.begin(), l1.end());
        auto l2 = layer2.parameters();
        params.insert(params.end(), l2.begin(), l2.end());
        return params;
    }
};

int main() {
    simplenet::Device dev = simplenet::Device::cuda(0);

    Model model(2, 5, dev);

    simplenet::Tensorf input({1, 2});
    input.linspace(1, 2);
    input.to_(dev);

    simplenet::Tensorf target({1, 5});
    target.linspace(1, 5);
    target.to_(dev);

    simplenet::neural_network::optimizers::SGD optim(model.parameters(), 0.1);

    for (int i = 0; i < 10; i++) {
        optim.zero_grad();

        auto pred = model.forward({input});
        auto actual_node = simplenet::Node<simplenet::Tensorf>::make_node(target);
        auto loss = simplenet::neural_network::loss_functions::l1_loss(actual_node, pred);

        simplenet::autogradient::backward(loss);
        optim.step();

        std::cout << "Loss: " << loss->val << std::endl;
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
* Templated `Tensor<T>` class with dtype aliases (`TensorD`, `Tensorf`, `TensorI`)
* Reverse-mode automatic differentiation, generic over scalar `double` and any `Tensor<T>` (via the `is_tensor_v` trait)
* Neural network module system: Linear, ReLU, Sigmoid, LeakyReLU, Tanh
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
    * Apply corresponding modifications to multiplication in `autogradient.h`.
* **Caching:** Implement tensor caching to reduce memory usage.
* **Convolution:** Implement tensor convolution support.

### Autogradient/Autodiff & Neural Network Module
* **Autodiff Engine:** Implement for activation functions (Unary and Binary ops).
* **Sequential Class:** Create a wrapper to stack layers.
* **Loss Functions:**
    * Cross Entropy Loss
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
* **Advanced Features:**
    * Mixed precision training -- **High Importance**
    * Dropout.
    * He initialization (to supplement current Xavier default).
* **Cleanup:** Removing Eigen operations **(Long-term)**.

### Compatibility and Containerization
* Provide a Dockerfile for building the project.
* Add compatibility for `clang` and `msvc` compilers.

---

## Contributing

This repository is open to contributions. Please make an issue before submitting a pull request.

### Getting Started

1. Fork and clone the repository (with submodules):
   ```bash
   git clone --recursive https://github.com/GodRishUniverse/SimpleNet.git
   cd SimpleNet
   ```

2. Create a build directory and build:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. Run the test driver to verify your build:
   ```bash
   ./runchecks/runchecks
   ```

4. Run unit tests:
   ```bash
   cd .. && ./unit_tests.sh
   ```

### Making Changes

**Adding a new CUDA kernel:**

1. Create `your_kernel.cu` and `your_kernel.cuh` in `simplenet/cuda/kernels/`.
2. In the `.cuh` file, declare the kernel function and the `launch_*` host wrapper. Use the existing pattern with `INSTANTIATE_*` macros for template instantiation.
3. In the `.cu` file, implement the `__global__` kernel and the `launch_*` function. Follow the existing pattern for stream management (create a stream if `nullptr` is passed, synchronize and destroy if owned).
4. Add forward declarations in `simplenet/cuda/includes/kernel_links.h` so host code can call your launcher.
5. Add the `.cu` file to `simplenet/CMakeLists.txt` in the source list.

**Adding a new layer/activation:**

1. Create your class inheriting from `simplenet::neural_network::Module` in `simplenet/activation_functions/`.
2. Implement `forward()`, `parameters()`, `get_detached_value()`, and the `operator()` overloads.
3. Include it in `simplenet/simplenet.h` if it should be part of the public API.

**Adding a new loss function:**

1. Add your function in `simplenet/loss_functions/loss.h` following the existing pattern.
2. It should take `shared_ptr<Node<Tensor>>` arguments and return the same. Use the autodiff operators (`+`, `-`, `*`, `/`, `hadamard`, etc.) so gradients flow through automatically.

**Adding a new optimizer:**

1. Inherit from `simplenet::neural_network::optimizers::Optimizer` in `simplenet/optimizers/optimizers.h`.
2. Implement `step()` and `zero_grad()`.
3. Add the implementation in `optimizers.cpp`.

**Adding new Tensor operations:**

1. If it's a host-side utility, add it to the appropriate file under `simplenet/tensor/utils/`.
2. If it needs a CUDA kernel, follow the kernel instructions above and add the host dispatch in `cuda_kernels.h/.cpp`.
3. Add the public method to `Tensor.h` / `Tensor.cpp`.

### Build Configuration

The project uses CMake with the following defaults (see top-level `CMakeLists.txt`):
- C++ standard: C++20
- CUDA standard: C++17
- Compiler: GCC 14 (`/usr/bin/gcc-14`)
- CUDA Toolkit: 13.0

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
>
