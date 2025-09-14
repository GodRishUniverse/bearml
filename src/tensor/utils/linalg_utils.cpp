#include "linalg_utils.h"
#include "tensor/Tensor.h"
#include <cstddef>

namespace simplenet {
    namespace linear_algebra {

        // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
        Tensor batchedMatMul(const Tensor& a, const Tensor& b){
            // TODO: figure out how batched mat mul will be implemented and use Eigen for mul as it does not have an inbuilt batch mat mul
            //
            // can do normal mat mul for elements
            std::vector<int> a_size = a.getShape();
            std::vector<int> b_size = b.getShape();

            // [a1, a2, ..., an, am] * [b1, b2, ..., bm, bk]
            // We need to treat these as batches of
            // batches_(a1*a2*...*)[an, am] * batches_(b1*b2*...*)[bm, bk] work on so each version of [an,am]*[bm*bk] are multiplied with
            // An example of a working version is alson
            // [2,1,4,5] * [2,3,5,6] -> [2,3,4,6] as the output shape - broadcasting done
            // Get matrix dimensions (last two dimensions from each Tensor)
            int a_rows = a_size[a_size.size() - 2];
            int a_cols = a_size[a_size.size() - 1];
            int b_rows = b_size[b_size.size() - 2];
            int b_cols = b_size[b_size.size() - 1];

            if (a_cols != b_rows){
                throw std::invalid_argument("SHAPES are incompatible for batched matmul: "+ std::to_string(a_cols) + " != " + std::to_string(b_rows));
            }

            // Compute broadcast shape for batch dimensions
            std::vector<int> a_batch_dims(a_size.begin(), a_size.end() - 2);
            std::vector<int> b_batch_dims(b_size.begin(), b_size.end() - 2);

            std::vector<int> batch_shape;
            if (a_batch_dims.empty()) {
                batch_shape = b_batch_dims;
            } else if (b_batch_dims.empty()) {
                batch_shape = a_batch_dims;
            } else {
                // Both have batch dimensions - broadcast them
                batch_shape = utils::computeBroadcastShape(a_batch_dims, b_batch_dims);
            }


            // Create output shape: batch_shape + [a_rows, b_cols]
            std::vector<int> output_shape = batch_shape;
            output_shape.push_back(a_rows);
            output_shape.push_back(b_cols);

            Tensor result(output_shape); // we have our result here

            // Create broadcast views for the full shapes (including matrix dims)
            std::vector<int> full_a_shape = batch_shape;
            full_a_shape.push_back(a_rows);
            full_a_shape.push_back(a_cols);

            std::vector<int> full_b_shape = batch_shape;
            full_b_shape.push_back(b_rows);
            full_b_shape.push_back(b_cols);

            Tensor a_view = Tensor::makeBroadcastView(a, full_a_shape);
            Tensor b_view = Tensor::makeBroadcastView(b, full_b_shape);

            // Perform batched matrix multiplication
            ll matrix_size_a = a_rows * a_cols;
            ll matrix_size_b = b_rows * b_cols;
            ll matrix_size_result = a_rows * b_cols;


            ll total_batch_size {1};
            for (int i : batch_shape){
                total_batch_size*=i;
            }
            total_batch_size = (total_batch_size>0) ? total_batch_size : 1;

            // loop over the total batch size -
            for (ll batch_index = 0; batch_index<total_batch_size; batch_index++){
                // now since we have the batch index with us we will be using the batch_coordinates
                std::vector<int> batch_coords(batch_shape.size());
                ll tmp = batch_index; // get the current batch index
                // HERE we convert our current batched index in the total to the coordinates in the actual tensor
                for (int d = batch_shape.size() - 1; d >= 0; --d) {
                    if (batch_shape[d] > 0) {
                        batch_coords[d] = tmp % batch_shape[d];
                        tmp /= batch_shape[d];
                    }
                }

                // Calculate offsets for this batch
                ll offset_a = 0, offset_b = 0;
                for (size_t d = 0; d < batch_shape.size(); ++d) {
                    offset_a += batch_coords[d] * a_view.strides[d]; // if stride is zero somewhere in the broadcasted view then the computations make use of the same memory - no copies done
                    offset_b += batch_coords[d] * b_view.strides[d]; // if stride is zero somewhere in the broadcasted view then the computations make use of the same memory - no copies done
                }

                // Get pointers to the matrices for this batch
                double* mat_a_ptr = a_view.data + offset_a;
                double* mat_b_ptr = b_view.data + offset_b;
                double* result_ptr = result.data + batch_index * matrix_size_result;

                using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


                // Use Eigen for the actual matrix multiplication - actual batched mat mul
                Eigen::Map<const MatrixRowMajor> mat_a(mat_a_ptr, a_rows, a_cols);
                Eigen::Map<const MatrixRowMajor> mat_b(mat_b_ptr, b_rows, b_cols);
                Eigen::Map<MatrixRowMajor> result_mat(result_ptr, a_rows, b_cols);

                result_mat =( mat_a * mat_b);
            }

            return result;
        }




        // Maybe I was thinking too much as this works
        Tensor reduce(const Tensor& a, std::vector<int>& afterShape){
            Tensor b = a;
            while (b.getShape().size() > afterShape.size()){
                b = b.sum(0, false); // we dont keep the dims
            }
            // Now we compare with the already existing values
            for (size_t i = 0; i<b.getShape().size(); i++){
                if (b.getShape()[i] != afterShape[i]){
                    b =b.sum(i, true); // we keep the dims
                }
            }
            return b;
        }





        // // Opposite of broadcasting - NOT A VIEW OPERATION
        // Tensor reduce(const Tensor &t, const std::vector<int>& targetShape) {

        //     //  NEED TO FIGURE OUT AN INVALID REDUCTION

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
