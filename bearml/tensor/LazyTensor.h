// We will create a tensor class (will see if we inherit from Tensor class or not) to be compiled
// TODO: Help with graph keeping and operator fusing of ops
// TODO: Can use XLA for faster execution as everything is precompiled
// TODO: need to do more research into XLA/MLIR and ML compilers and how they can be used to compile tensor ops

#pragma once
#include "Tensor.h"
#include "operators/ops.h"
#include <memory>
#include <vector>

//
//
// These notes were added to help me build intuition but I think doing it froom scratch and barebones approach is better and so see `ir.h`
//
//
//
// On working on that ->
// - We need to compute the forward graph here (do not compute or allocate anything)
// - then when we have the graph, we can use materialize to compute the actual tensor values -
//   - For materialize, we need to also make sure that we memoize so that each node caches its output and we don't recompute it
//   - another thing is we compute the max memory usage of the graph and then allocate memory once like ggml does so that we don't allocate memory for each node
//      - this ensures that we don't allocate memory for each node and instead allocate memory once for the entire graph
//      - sizing the arena once isn't enough on its own - still need a liveness pass (like register allocation) over
//        the topological order so nodes whose consumers are all done can hand their slot back for reuse, otherwise
//        "one big alloc" just becomes "one alloc sized to the sum of every node" instead of the peak
// - shape/dtype/device for an OP node has to be computed at construction time, not at materialize time - this is what
//   lets the memory-planning pass above run before any data exists, and it fails fast on shape mismatches too
// - broadcast/transpose/slice should be metadata-only nodes (no compute), same split Tensor.h already has between
//   makeBroadcastView/makeSliceView/makeStrideView (owns_data=false) and a real elementwise op - otherwise every
//   broadcast forces a materialize and fusion does not see through it
// - memoizing materialize() only avoids recompute when the SAME LazyTensorNode object is reached twice while
//   walking the graph (e.g. d = a+b and e = a+c both point at the same `a` node - `a` gets materialized once,
//   both d and e reuse it). It does NOT catch the case where two DIFFERENT node objects were built for the same
//   computation - e.g. `auto b = x*2; auto c = x*2;` builds two separate OP nodes with op_code=OP_MUL and the
//   same input `x`, and nothing above tells them apart, so both get materialized separately even though they'd
//   produce identical output. Catching that second case means, before materializing, walking the graph once and
//   grouping nodes by (op_code, the identities of their input nodes) - any two nodes with the same op_code over
//   the same input node objects are computing the same thing, so rewrite all but one to just point at that one
// - recursive materialize() (each node's materialize calls materialize() on its inputs, which calls materialize()
//   on their inputs, ...) is fine for correctness but a long chain of ops (e.g. 50 layers deep, one op per layer)
//   turns into 50 nested C++ function calls before the first one returns, which risks a stack overflow. The fix:
//   first build a flat list of every node in dependency order (every node appears in the list only after all of
//   its inputs already appear - the same ordering the liveness/arena pass needs), then materialize by walking
//   that list with a plain for-loop instead of letting each node call the next one recursively
// - in-place ops (+=, -= on Tensor) mutate the tensor's storage directly instead of producing a new value - if
//   that ever gets exposed on LazyTensor, a node's cached materialize() result can be silently invalidated after
//   the fact: e.g. node C caches the value of `a` at the moment it ran, then something does `a += 1` afterward -
//   C's cache is now wrong but nothing tells C to recompute, and the arena's liveness accounting (which assumed
//   a node's output stops changing once written) is wrong too. Simplest fix is to just not allow in-place ops to
//   touch anything a LazyTensorNode has already captured as an input
// - some LazyTensor accessors will need real data to answer (get() has to return an actual number, printing a
//   tensor has to show actual values) - each of those has to call materialize() first. That's fine if it happens
//   deliberately (e.g. once, at the very end of a training step, to log a loss value). It defeats the whole point
//   of laziness if it happens by accident - e.g. printing a LazyTensor for debugging inside a loop forces that
//   node and everything upstream of it to materialize on every single iteration, which means nothing ever gets a
//   chance to batch multiple ops together before running them
//

namespace bearml {

    // Uh Idk I need to design this properly
    // utility enum to distinguish between tensor nodes and op nodes
    enum class LazyTensorNodeType {
        TENSOR,
        OP
    };

    // TODO: check how to implement this and design this (the graph)
    template<typename T>
    struct LazyTensorMetadata{
        std::vector<int> shape; // will be used to store the mem alloc needed
        std::vector<int> strides;
        Device device;
        size_t data_offset;
        bool is_sliced_view;
    };

    // why not just make the graph and then call hip/cuda graphs directly and split the hip/cuda graphs where the ops dont exist (like we dont have kernels)??
    template <typename T>
    class LazyTensorNode {
        public:
            // we need incoming op and output op to know how to fuse
            LazyTensorNodeType node_type; // it can either be a tensor node or an op node
            std::string node_name; // the name of the node (for debugging purposes) - like can be variable name (if tensor) or op name ("add_1", "mul_2", etc.) - numbers are in topological order
            OP_Code op_code = OP_Code::NO_OP ; // all nodes are initially NO_OP - only set once node_type is known
            // TODO: I also need to extend my ops enum
            std::vector<std::shared_ptr<LazyTensorNode<T>>> inputs; // get the inputs to this op so that we can materialize it
            LazyTensorMetadata<T> metadata;

            //
            Tensor<T> materialize(); // materialize means to compute the tensor from the inputs
    };
}
