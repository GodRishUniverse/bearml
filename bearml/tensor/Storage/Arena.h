#pragma once
#include "Storage.h"
#include "devices/device_type.h"
#include <memory>

namespace bearml {

    // arena allocators are allocators that remove the malloc/free calls by allocating a big chunk once
    // and using offsets and free spaces via code
    // This avoids the complex malloc/free calls (memory ops are VERY VERY EXPENSIVE)
    // Our lazy tensor will use this - maybe tensor.h will also evolve (thinking about this)
    class Arena {
        private:
            std::shared_ptr<bearml::Storage> buffer; // one big allocation
            // TODO: remove offset and remaining_free_after_offset - the planner reuses slots, so a running total rejects valid plans
            size_t offset;
            size_t remaining_free_after_offset;
            // need a way to quickly check free storage places
            // TODO: free-space tracking belongs in the memory planner (plan_memory), not here - drop the note above
            // TODO: store an alignment (256 on CUDA, 64 on CPU) for slice() to check offsets against
        public:
            Arena(size_t bytes, const bearml::Device& device);
            std::shared_ptr<bearml::Storage> slice(size_t byte_offset, size_t nbytes) ; // return a chunk of free memory of this size
            // TODO: make slice() const once the counters are gone
    };

}
