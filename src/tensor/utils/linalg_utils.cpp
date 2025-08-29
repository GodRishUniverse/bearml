#include "linalg_utils.h"
#include "tensor/Tensor.h"

namespace simplenet {
    namespace linear_algebra {




        // // Opposite of broadcasting - NOT A VIEW OPERATION
        // Tensor reduce(const Tensor &t, const std::vector<int>& targetShape) {

        //     // TODO: NEED TO FIGURE OUT AN INVALID REDUCTION

        //     // Idea -   First pass: analyze the shapes and build a "recipe" of operations
        //     //          Second pass: execute the recipe
        //     // Some questions that helped me to think more about -
        //     //      Which dimensions need to be flattened together?
        //     //      Which dimensions need to be summed (with keepdims)?
        //     //      What order should these operations happen in?
        //     //      How would I search this?
        //     std::vector<Operations::ReductionOp> operations;
        //     std::vector<int> currentShape = t.getShape();

        //     // 3 Cases to consider here ->
        //     // Fall through cases exist here
        //     //  3. If current shape has more size then target shape-> flatten the v shape till the same size
        //     //  2. If the index position values differ then basically to a sum in the currentShape onto the target shape
        //     //      - if targetShape has higher value here then target is copied
        //     //  1. if the index and values on the index are the same leave as it is.

        //     // 3. If current shape has more size then target shape-> flatten the v shape till the same size
        //     if (currentShape.size() > targetShape.size()){
        //         int dimsToFlatten = currentShape.size() - targetShape.size(); // we flatten v till a point

        //         operations.push_back({Operations::ReductionOp::FLATTEN, 0, dimsToFlatten, false});  // v.flatten<void>(0, v.shape.size()- targetShape.size(), false); // we do not want to keep the dimensions
        //         // Update shape after flattening for further analysis
        //         int flattenedSize = 1;
        //         for (int i = 0; i < dimsToFlatten; i++) {
        //             flattenedSize *= currentShape[i];
        //         }
        //         currentShape.erase(currentShape.begin(), currentShape.begin() + dimsToFlatten);
        //         currentShape.insert(currentShape.begin(), flattenedSize);
        //     }

        //     // compare dimensions and add sum operations where needed
        //     for (int i = currentShape.size() - 1; i >= 0; i--) {
        //         if (currentShape[i] != targetShape[i]) {
        //             if (targetShape[i] == 1) {
        //                 operations.push_back({Operations::ReductionOp::SUM, i, i, true});
        //                 currentShape[i] = 1;
        //             } else {
        //                 throw std::invalid_argument("Incompatible shapes for reduce operation");
        //             }
        //         }
        //     }

        //     // iterate from right to left to see which shapes match and which dont- THING to figure out
        //     // ~~PROBLEM: sumation will change shapes so need to figure out how to do it efficiently~~
        //     // Solution - this is not a view operation and so memory reallocation will have to be done as there is no way to manipulate strides here as far as my research shows
        //     // WILL use our sum function created below
        //     Tensor v = t;
        //     // DO all operations here
        //     for (const auto& op : operations) {
        //         if (op.type == Operations::ReductionOp::FLATTEN) {
        //             v.flatten_inplace(op.startDim, op.endDim, op.keepdims);
        //         } else if (op.type == Operations::ReductionOp::SUM) {
        //             v = v.sum(op.startDim, op.keepdims);
        //         }
        //     }
        //     return v;
        // }

        // // The idea for a reduction is summation and flattening so this just makes it explicit
        // Tensor flatten_and_sum_to_shape(const Tensor &t, const std::vector<int>& targetShape){
        //     return reduce(t, targetShape);
        // }
    }
}
