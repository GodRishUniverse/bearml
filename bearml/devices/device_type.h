#pragma once
#include <string>
#include <stdexcept>

namespace bearml {

    // Our enum for the devices that we will have
    enum class DeviceType {
        CPU,
        CUDA,
        HIP // HIP is the AMD equivalent of CUDA - CUDA and HIP should be interchangeable cause of AMD's hipify tool
    };

    class Device {
        public:
            DeviceType type;
            int device_id;  // if we have multiple gpus

            Device() : type(DeviceType::CPU), device_id(-1) {}
            Device(DeviceType t, int id = 0) : type(t), device_id(id) {}

            static Device cpu() { return Device(DeviceType::CPU, -1); }
            static Device cuda(int id = 0) { return Device(DeviceType::CUDA, id); }
            static Device hip(int id = 0) { return Device(DeviceType::HIP, id); }


            bool is_cpu() const { return type == DeviceType::CPU; }
            bool is_cuda() const { return type == DeviceType::CUDA; }
            bool is_hip() const { return type == DeviceType::HIP; }

            std::string to_string() const {
                if (is_cpu()) return "cpu";
                if (is_hip()) return "hip:" + std::to_string(device_id);
                return "cuda:" + std::to_string(device_id);
            }

            bool operator==(const Device& other) const {
                // we dont care about device_id for CPU devices
                if (type == DeviceType::CPU && other.type == DeviceType::CPU) {
                    return true;
                }
                if (type == DeviceType::CUDA && other.type == DeviceType::CUDA) {
                    return device_id == other.device_id;
                }
                if (type == DeviceType::HIP && other.type == DeviceType::HIP) {
                    return device_id == other.device_id;
                }
                return false;
            }

            bool operator!=(const Device& other) const {
                return !(*this == other);
            }
    };

    // Macro functions -> https://stackoverflow.com/questions/163365/how-do-i-make-a-c-macro-behave-like-a-function
    // error checking macro
    #ifndef CUDA_CHECK_MACRO
    #define CUDA_CHECK_MACRO
    #if defined(BEARML_USE_CUDA)
        #define CUDA_CHECK(call) \
            do { \
                cudaError_t error = call; \
                if (error != cudaSuccess) { \
                    throw std::runtime_error( \
                        std::string("CUDA error at ") + __FILE__ + ":" + \
                        std::to_string(__LINE__) + " - " + \
                        cudaGetErrorString(error)); \
                } \
            } while(0)
    #else
        // CPU-only build: deliberately does NOT expand `call`, so the wrapped
        // cudaMalloc/cudaMemcpy/... symbols are never referenced.
        #define CUDA_CHECK(call) \
            throw std::runtime_error( \
                std::string("CUDA op at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + \
                " invoked but BearML built without CUDA (BEARML_USE_CUDA=OFF)")
    #endif
    #endif
}
