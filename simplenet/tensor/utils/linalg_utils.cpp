#include "linalg_utils.h"
#include "tensor/Tensor.h"
#include <cstddef>
#include <stdexcept>

namespace simplenet {
    namespace linear_algebra {

        // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
        Tensor batchedMatMul(const Tensor& a, const Tensor& b){
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

            if (a.device != b.device) {
                throw std::invalid_argument("Tensors must be on the same device");
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

            Tensor result(output_shape, a.device); // we have our result here

            // Create broadcast views for the full shapes (including matrix dims)
            std::vector<int> full_a_shape = batch_shape;
            full_a_shape.push_back(a_rows);
            full_a_shape.push_back(a_cols);

            std::vector<int> full_b_shape = batch_shape;
            full_b_shape.push_back(b_rows);
            full_b_shape.push_back(b_cols);

            Tensor a_view = Tensor::makeBroadcastView(a, full_a_shape);
            Tensor b_view = Tensor::makeBroadcastView(b, full_b_shape);

            ll total_batch_size {1};
            for (int i : batch_shape){
                total_batch_size*=i;
            }
            total_batch_size = (total_batch_size>0) ? total_batch_size : 1;

            // CUDA path
            if (a.device == DeviceType::CUDA) {
                // Use broadcast kernel with strides computed from the views
                // We only need the batch strides (not the matrix dim strides)
                std::vector<int64_t> batch_strides_a(a_view.strides.begin(),
                    a_view.strides.begin() + batch_shape.size());
                std::vector<int64_t> batch_strides_b(b_view.strides.begin(),
                    b_view.strides.begin() + batch_shape.size());

                cuda::launch_gemm_broadcasted<double>(
                    a.data, b.data, result.data,
                    a_rows, a_cols, b_cols,
                    1.0, 0.0,
                    &batch_shape,
                    batch_shape.size(),
                    &batch_strides_a,
                    &batch_strides_b,
                    total_batch_size,
                    nullptr
                );

                return result;
            }

            // CPU path
            ll matrix_size_result = a_rows * b_cols;

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



        // TODO: write CUDA kernel
        // Maybe I was thinking too much as this works
        Tensor reduce(const Tensor& a, std::vector<int>& afterShape, reductions::ReductionOps op){
            Tensor b = a;
            while (b.getShape().size() > afterShape.size()){
                b = b.accumulate(0, op, false); // we dont keep the dims
            }
            // Now we compare with the already existing values
            for (size_t i = 0; i<b.getShape().size(); i++){
                if (b.getShape()[i] != afterShape[i]){
                    b =b.accumulate(i, op, true); // we keep the dims
                }
            }
            return b;
        }

        Tensor hadamard(const Tensor &a, const Tensor &other) {
            if (a.shape != other.shape){
                throw std::invalid_argument("Tensors must have the same shape");
            }

            if (a.device != other.device) {
                throw std::invalid_argument("Tensors must be on the same device");
            }

            // CUDA
            if (a.device == DeviceType::CUDA) {
                Tensor C(a.shape, a.device);

                cuda::launch_elementwise_broadcast<double>(a.data, other.data, C.data, a.getStrides(), other.getStrides(), C.getShape(), OP_Code::OP_MUL);
                return C;
            }

            Tensor result(a.shape, a.device);
            for (ll i = 0; i < a.sizeOfTensor(); i++){
                result.data[i] = a.data[i] * other.data[i];
            }
            return result;
        }


        // -----------------------------------------------Operations helpful in mask generations------------------

        Tensor compare(const Tensor& a, const Tensor& b, CompareOp op, double true_val, double false_val) {
            if (a.getShape() != b.getShape()) {
                throw std::runtime_error("In compare and shapes don't match");
            }
            Tensor result(a.getShape(), a.device);

            if (a.device != b.device) {
                throw std::invalid_argument("Tensors must be on the same device");
            }


            if (a.device.type == DeviceType::CUDA) {
                switch (op) {
                    case CompareOp::GT: // greater than
                        simplenet::cuda::launch_comparison_kernel<double>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::GE: // greater than or equal to
                        simplenet::cuda::launch_comparison_kernel<double>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::LT: // less than
                        simplenet::cuda::launch_comparison_kernel<double>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::LE: // less than or equal to
                        simplenet::cuda::launch_comparison_kernel<double>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::EQ: // equal to
                        simplenet::cuda::launch_comparison_kernel<double>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::NE: // not equal to
                        simplenet::cuda::launch_comparison_kernel<double>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    default:
                        throw std::invalid_argument("Invalid compare op");
                }
            } else {
                for (size_t i = 0; i < a.sizeOfTensor(); i++) {
                    switch (op) {
                        case CompareOp::GT: // greater than
                            result.data[i] = (a.data[i] > b.data[i]) ? true_val : false_val;
                            break;
                        case CompareOp::GE: // greater than or equal to
                            result.data[i] = (a.data[i] >= b.data[i]) ? true_val : false_val;
                            break;
                        case CompareOp::LT: // less than
                            result.data[i] = (a.data[i] < b.data[i]) ? true_val : false_val;
                            break;
                        case CompareOp::LE: // less than or equal to
                            result.data[i] = (a.data[i] <= b.data[i]) ? true_val : false_val;
                            break;
                        case CompareOp::EQ: // equal to
                            result.data[i] = ((std::abs(a.data[i] - b.data[i]) < 1e-12)) ? true_val : false_val;
                            break;
                        case CompareOp::NE: // not equal to
                            result.data[i] = ((std::abs(a.data[i] - b.data[i]) >= 1e-12)) ? true_val : false_val;
                            break;
                        default:
                            throw std::invalid_argument("Invalid compare op");
                    }
                }
            }
            return result;
        }

        Tensor mask_of_greater_than_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val) {
            return compare(first, other, CompareOp::GE, first_val, second_val);
        }

        Tensor mask_of_greater_than(const Tensor& first, const Tensor& other,  double first_val, double second_val) {
            return compare(first, other, CompareOp::GT, first_val, second_val);
        }

        Tensor mask_of_less_than_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val) {
            return compare(first, other, CompareOp::LE, first_val, second_val);
        }

        Tensor mask_of_less_than(const Tensor& first, const Tensor& other,  double first_val, double second_val) {
            return compare(first, other, CompareOp::LT, first_val, second_val);
        }

        Tensor mask_of_equal_to(const Tensor& first, const Tensor& other,  double first_val, double second_val){
            return compare(first, other, CompareOp::EQ, first_val, second_val);
        }



        // Double and Tensor
        Tensor mask_of_greater_than_equal_to(double first, const Tensor& other,  double first_val, double second_val) {

            Tensor result(other.getShape(), other.device);

            for (size_t i = 0; i < other.sizeOfTensor(); i++) {
                result.data[i] = (first >= other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_greater_than(double first, const Tensor& other,  double first_val, double second_val){
            Tensor result(other.getShape(), other.device);

            for (size_t i = 0; i < other.sizeOfTensor(); i++) {
                result.data[i] = (first > other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_less_than_equal_to(double first, const Tensor& other,  double first_val, double second_val){
            Tensor result(other.getShape(), other.device);

            for (size_t i = 0; i <  other.sizeOfTensor(); i++) {
                result.data[i] = (first <= other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_less_than(double first, const Tensor& other,  double first_val, double second_val){
            Tensor result(other.getShape(), other.device);

            for (size_t i = 0; i <  other.sizeOfTensor(); i++) {
                result.data[i] = (first < other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_equal_to(double first, const Tensor& other,  double first_val, double second_val){
           Tensor result(other.getShape(), other.device);
            for (size_t i = 0; i < other.sizeOfTensor(); ++i) {
                result.data[i] = (std::abs(first - other.data[i]) < 1e-12) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        // Tensor and Double
        Tensor mask_of_greater_than_equal_to(const Tensor& first, double other,  double first_val, double second_val){
            Tensor result(first.getShape(), first.device);

            for (size_t i = 0; i < first.sizeOfTensor(); i++) {
                result.data[i] = (first.data[i] >= other) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_greater_than(const Tensor& first, double other,  double first_val, double second_val) {
            Tensor result(first.getShape(), first.device);
            for (size_t i = 0; i < first.sizeOfTensor(); ++i) {
                result.data[i] = (first.data[i] > other) ? first_val: second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_less_than_equal_to(const Tensor& first, double other,  double first_val, double second_val){
            Tensor result(first.getShape(), first.device);

            for (size_t i = 0; i < first.sizeOfTensor(); i++) {
                result.data[i] = (first.data[i] <= other) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_less_than(const Tensor& first, double other,  double first_val, double second_val){
            Tensor result(first.getShape(), first.device);

            for (size_t i = 0; i < first.sizeOfTensor(); i++) {
                result.data[i] = (first.data[i] < other) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        Tensor mask_of_equal_to(const Tensor& first, double other,  double first_val, double second_val){
            Tensor result(first.getShape(), first.device);
            for (size_t i = 0; i < first.sizeOfTensor(); ++i) {
                result.data[i] = (std::abs(first.data[i] - other) < 1e-12) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        // TODO: templatize
        Tensor sign(const Tensor& a){
            Tensor result(a.getShape(), a.device);

            if (a.getDevice().type == DeviceType::CPU) {
                for (size_t i = 0; i < a.sizeOfTensor(); ++i) {
                    result.data[i] = (std::abs(a.data[i] - 0.0) < 1e-12) ? 0.0 : ((a.data[i] < 0.0) ? -1.0 : 1.0);
                }
                return result;
            }

            // CUDA implementation
            cuda::launch_sign_contiguous<double>(a.data, result.data, a.getShape());

            return result;
        }

        // inverse matrix
        Tensor inverse(const Tensor& a) {
            // inverse is only defined for square matrices
            std::vector<int> a_shape = a.getShape();

            if (a_shape.size() < 2) {
                throw std::runtime_error("inverse: only square matrices are supported. Passed in a vector or a scalar.");
            }

            if (a_shape[a_shape.size() - 1] != a_shape[a_shape.size() - 2]) {
                throw std::runtime_error("inverse: only square matrices are supported");
            }

            Tensor result = a; // copy the input tensor

            if (a.getDevice().type == DeviceType::CPU) {
                using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

                for (int batch = 0; batch < a.sizeOfTensor() / (a_shape[a_shape.size() - 1] * a_shape[a_shape.size() - 2]); ++batch) {
                    int row = a_shape[a_shape.size() - 1];
                    int col = a_shape[a_shape.size() - 2];
                    Eigen::Map<MatrixRowMajor> matrix(result.data + batch * row * col, row, col); // get the matrix for this batch
                    Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
                    if (!lu.isInvertible()) {
                        throw std::runtime_error("Matrix is not invertible");
                    }
                    matrix = lu.inverse(); // map handles the memory copy back to result.data
                }
            }

            // TODO: CUDA implementation

            return result;
        }


        // Example for im2col_2d is as follows
        // Input (4×4):            Patches extracted (2×2 = 4 positions):
        //  1  2  3  4             Position (0,0): [1,2,3, 5,6,7, 9,10,11]
        //  5  6  7  8             Position (0,1): [2,3,4, 6,7,8, 10,11,12]
        //  9 10 11 12             Position (1,0): [5,6,7, 9,10,11, 13,14,15]
        // 13 14 15 16             Position (1,1): [6,7,8, 10,11,12, 14,15,16]

        // im2col matrix (9 rows × 4 columns):
        // [ 1   2   5   6 ]
        // [ 2   3   6   7 ]
        // [ 3   4   7   8 ]
        // [ 5   6   9  10 ]
        // [ 6   7  10  11 ]
        // [ 7   8  11  12 ]
        // [ 9  10  13  14 ]
        // [10  11  14  15 ]
        // [11  12  15  16 ]

        // TODO: optimize this O(n^6) loop and implement CUDA version
        Tensor im2col_2d(Tensor& a, int kernel_size, int stride, int padding, int dilation){

            if (dilation < 1) {
                throw std::runtime_error("Dilation must be greater than or equal to 1");
            }

            if (padding < 0) {
                throw std::runtime_error("Padding must be greater than or equal to 0");
            }

            std::vector<int> shape = a.getShape();

            size_t num_channels = (shape.size() >= 3) ? shape[shape.size()-3] : 1;
            size_t height = (shape.size() >= 2) ? shape[shape.size()-2] : 1;
            size_t width = shape[shape.size()-1];
            size_t batch_size = a.sizeOfTensor() / (num_channels * height * width);

            int H = static_cast<int>(height);
            int W = static_cast<int>(width);
            int H_out = (H + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1;
            int W_out = (W + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1;

            if (H_out <= 0 || W_out <= 0) {
                throw std::runtime_error("Invalid output dimensions - kernel too large for input");
            }

            double padding_value = 0.0;

            // patch matrix shape is num_channels*K*K, H_out*W_out

            int num_rows_in_patch = num_channels*kernel_size*kernel_size; // C_in * K * K
            int patch_size = H_out * W_out; // H_out * W_out

            std::vector<int> patch_shape = {batch_size, num_rows_in_patch, patch_size};
            Tensor patch(patch_shape, a.getDevice());

            // TODO: implement for CUDA
            // b,c,h,w -> b,c*k*k, h_out*w_out
            // TODO test this out
            for (size_t b = 0; b < batch_size; ++b) {
                for (int channels = 0; channels < num_channels; ++channels) {
                    // kernel size loop
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            // this already knows the patch size and stride
                            for (int h = 0; h < H_out; ++h) {
                                for (int w = 0; w < W_out; ++w) {
                                    // get the value from the input tensor by indexing
                                    int h_in = h * stride - padding + kh*dilation; // h*stride gives the global row index and then -1*padding ensures we start from the correct row
                                    int w_in = w * stride - padding + kw*dilation;
                                    double value = padding_value;
                                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                        value = a({b, channels, h_in, w_in});
                                    }
                                    patch.set(value, {b, channels*kernel_size*kernel_size + kh*kernel_size + kw, h*W_out + w});
                                }
                            }
                        }
                    }
                }
            }

            return patch;

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
