#pragma once

#include <cstddef>
#include <memory>
#include "devices/device_allocator.h"

namespace bearml {




    class Storage {
    private:
           void * data_ = nullptr; // untyped pointer to the data
           size_t num_bytes_;
           Device device_;
           std::unique_ptr<DeviceAllocator> allocator_; // the device allocator we will be using for moving -> we only want a unique allocator
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

        void* raw_data() const {
            return data_;
        }

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
