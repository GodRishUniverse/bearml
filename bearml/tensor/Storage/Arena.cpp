#include "Arena.h"
#include <stdexcept>

namespace bearml {

    Arena::Arena(size_t bytes, const Device& device):
        buffer(std::make_shared<Storage>(bytes, device)), offset(0), remaining_free_after_offset(bytes){
        // need a free memory inside the offseted
    }

    std::shared_ptr<Storage> Arena::slice(size_t byte_offset, size_t nbytes){
        // TODO: replace both checks with one: byte_offset > size || nbytes > size - byte_offset (no overflow), throw out_of_range
        // TODO: debug assert byte_offset % alignment == 0
        if (byte_offset>= buffer->number_of_bytes()){
            throw std::logic_error("Arena (mempool): Cannot have byte offset larger than buffer size");
        }

        if (nbytes>remaining_free_after_offset){
            throw std::out_of_range("Arena (mempool): Cannot allocate more memory than available");
        }

        // TODO: remove - slots are reused across lifetimes, so slice(0,60) twice in a 100-byte arena must succeed
        this->offset+=byte_offset;
        this->remaining_free_after_offset-=nbytes;

        return std::make_shared<Storage>(buffer, byte_offset, nbytes);
    }
}
