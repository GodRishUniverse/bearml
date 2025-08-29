
#include <iostream>

// Simple CUDA kernel that adds two arrays


__global__ void addKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      c[idx] = a[idx] + b[idx];
    }
  }
  
  int main() {
    int n = 1024;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
  
    // Allocate host memory
    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));
    c = (float *)malloc(n * sizeof(float));
  
    // Initialize host memory
    for (int i = 0; i < n; i++) {
      a[i] = i;
      b[i] = i * 2;
    }
  
    // Allocate device memory
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));
  
    // Copy host memory to device memory
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
  
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
  
    // Copy device memory back to host memory
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  
    // Print results
    for (int i = 0; i < n; i++) {
      std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }
  
    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  
    return 0;
  }
