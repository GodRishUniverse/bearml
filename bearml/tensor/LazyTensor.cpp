#include "LazyTensor.h"
namespace bearml {

    // basic constructor ()
    LazyTensorNode::LazyTensorNode(LazyTensorNodeType type, std::string name){
        node_type = type;
        node_name = name;
    }

    // Helper function for lazy tensor graph total memory usage - TODO: understand how graph will be passed in here
    void calculate_lazy_tensor_memory_usage() {

    }
}
