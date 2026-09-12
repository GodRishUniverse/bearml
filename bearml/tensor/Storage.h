#pragma once

#include <cstddef>
#include <memory>
#include "devices/device_allocator.h"

namespace bearml {

    // Owns one contiguous allocation on one device. Tensors hold a shared_ptr to it, so
    // views (broadcast/slice/permute) share the pointer instead of each tracking ownership.
    class Storage {
    private:
           void * data_ = nullptr; // untyped pointer to the data
           size_t num_bytes_;
           Device device_;
           // get_allocator() hands back a per-device singleton - so the uniqueness is guaranteed there
           DeviceAllocator* allocator_;
    public:
        Storage(size_t nbytes, const Device& device) :
            num_bytes_(nbytes),
            device_(device),
            allocator_(&get_allocator(device)) {

                  if (num_bytes_ > 0) {
                      data_ = allocator_->allocate(num_bytes_);
                  }
        }
        // no move or copy constructors as we only want shared ownership
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        Storage(Storage&&) = delete;
        Storage& operator=(Storage&&) = delete;

        ~Storage() {
            if (data_) {
                allocator_->deallocate(data_);
                data_ = nullptr;
            }
        }

        void* raw_data() const { return data_; }

        template <typename T>
        T* data() const {
            return static_cast<T*>(data_);
        }
        // getters
        size_t number_of_bytes() const { return num_bytes_; }
        const Device& device() const { return device_; }
        DeviceAllocator& allocator() const { return *allocator_; }

    };
}
