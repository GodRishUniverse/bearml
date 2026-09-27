#pragma once
#if defined(BEARML_USE_CUDA)
#include <cuda_runtime.h>
#endif
#include "device_type.h"
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <memory>



namespace bearml {

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

#if defined(BEARML_USE_CUDA)
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
#endif // BEARML_USE_CUDA

    // factory function
    // returns the type of allocator
    inline DeviceAllocator& get_allocator(const Device& device) {
        if (device.is_cpu()) {
            static CPUAllocator cpu_allocator;
            return cpu_allocator;
        } else {
#if defined(BEARML_USE_CUDA)
            static std::mutex mu; // static mutex so all threads see same mutex
            static std::unordered_map<int, std::unique_ptr<CUDAAllocator>> gpus; // static map so all threads see same gpus which have unique cuda allocators to them
            std::lock_guard<std::mutex> lock(mu); // in-built locking mechanism is cpp that ensures that we unlock after we return from this function (or exception)
            auto& slot = gpus[device.device_id];
            if (!slot) slot = std::make_unique<CUDAAllocator>(device.device_id); // if no allocator for this device, create one
            return *slot; // return appropriate allocator
#else
            throw std::runtime_error(
                "Requested a CUDA device but BearML was built without CUDA "
                "(BEARML_USE_CUDA=OFF). Rebuild with CUDA enabled.");
#endif
        }
    }

}
