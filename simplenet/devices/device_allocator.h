#pragma once
#if defined(SIMPLENET_USE_CUDA)
#include <cuda_runtime.h>
#endif
#include "device_type.h"
#include <cstring>
#include <stdexcept>

namespace simplenet {

    // abstract allocator class
    class DeviceAllocator {
    public:
        virtual ~DeviceAllocator() = default;
        virtual void* allocate(size_t bytes) = 0;
        virtual void deallocate(void* ptr) = 0;
        virtual void copy_to_device(void* destination, const void* source, size_t bytes) = 0; // copy to gpu
        virtual void copy_to_host(void* destination, const void* source, size_t bytes) = 0; // copy to cpu from gpu(s)
        virtual void copy_device_to_device(void* destination, const void* source, size_t bytes) = 0; // copy fron one gpu to another
        virtual Device device() const = 0;
    };

    // TODO: templatify for different data types -> float16, float32 and int8
    class CPUAllocator : public DeviceAllocator {
    public:
        void* allocate(size_t bytes) override {
            // Allocate raw, max-aligned bytes. Must NOT assume a double element type:
            // `new double[bytes/sizeof(double)]` floors to a multiple of 8 bytes and
            // under-allocates for element types like float (e.g. 5 floats = 20 bytes
            // would only get 16), corrupting the tail element.
            return ::operator new(bytes);
        }

        void deallocate(void* ptr) override {
            ::operator delete(ptr);
        }

        void copy_to_device(void* destination, const void* source, size_t bytes) override {
            std::memcpy(destination, source, bytes);
        }

        void copy_to_host(void* destination, const void* source, size_t bytes) override {
            std::memcpy(destination, source, bytes);
        }

        void copy_device_to_device(void* destination, const void* source, size_t bytes) override {
            std::memcpy(destination, source, bytes);
        }

        Device device() const override { return Device::cpu(); }
    };

#if defined(SIMPLENET_USE_CUDA)
    // CUDA Allocator
    class CUDAAllocator : public DeviceAllocator {
        private:
            int device_id;

        public:
            explicit CUDAAllocator(int device_id = 0) : device_id(device_id) {
                CUDA_CHECK(cudaSetDevice(this->device_id));
            }

            void* allocate(size_t bytes) override {
                void* ptr = nullptr;
                CUDA_CHECK(cudaSetDevice(this->device_id));
                CUDA_CHECK(cudaMalloc(&ptr, bytes));
                return ptr;
            }

            void deallocate(void* ptr) override {
                if (ptr) {
                    CUDA_CHECK(cudaSetDevice(this->device_id));
                    CUDA_CHECK(cudaDeviceSynchronize()); // sync before deletion
                    CUDA_CHECK(cudaFree(ptr));
                }
            }

            void copy_to_device(void* destination, const void* source, size_t bytes) override {
                CUDA_CHECK(cudaSetDevice(this->device_id));
                CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
            }

            void copy_to_host(void* destination, const void* source, size_t bytes) override {
                CUDA_CHECK(cudaSetDevice(this->device_id));
                CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaDeviceSynchronize()); // sync
            }

            void copy_device_to_device(void* destination, const void* source, size_t bytes) override {
                CUDA_CHECK(cudaSetDevice(this->device_id));
                CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice));
            }

            Device device() const override { return Device::cuda(this->device_id); }
    };
#endif // SIMPLENET_USE_CUDA

    // factory function
    // returns the type of allocator
    inline DeviceAllocator* get_allocator(const Device& device) {
        if (device.is_cpu()) {
            return new CPUAllocator();
        } else {
#if defined(SIMPLENET_USE_CUDA)
            return new CUDAAllocator(device.device_id);
#else
            throw std::runtime_error(
                "Requested a CUDA device but SimpleNet was built without CUDA "
                "(SIMPLENET_USE_CUDA=OFF). Rebuild with CUDA enabled.");
#endif
        }
    }

}
