#pragma once
// Template definitions for bearml::linear_algebra (declared in linalg_utils.h).
// Included from the bottom of tensor/Tensor.h once Tensor<T> is complete.
#include <cstddef>
#include <stdexcept>

namespace bearml {
    namespace linear_algebra {


        // THIS is where we will be doing the multiplication when the dimensions exceed the normal 2 of a matrix
        template<typename T>
        Tensor<T> batchedMatMul(const Tensor<T>& a, const Tensor<T>& b){
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

            // The broadcast GEMM kernel and the CPU Eigen RowMajor Map below both assume each
            // inner (M,K) / (K,N) matrix is laid out row-major contig. A transposed or permuted
            // operand would silently produce wrong results, so densify on entry. Operator*'s 2D
            // path (case 6) is layout-aware; once gemm_kernel_broadcast also takes per-operand
            // strides, this densify can drop.
            auto inner_is_row_major = [](const Tensor<T>& t){
                int nd = (int)t.getShape().size();
                if (nd < 2) return true;
                int K = t.getShape()[nd - 1];
                auto s = t.getStrides();
                return s[nd - 1] == 1 && s[nd - 2] == K;
            };
            // We avoid densifying when only the BATCH dims are broadcast/permuted (inner matrix
            // is still row-major contig in storage). makeBroadcastView further down still handles
            // those correctly via stride 0.
            Tensor<T> a_storage = inner_is_row_major(a) ? a : Tensor<T>::contiguous(a);
            Tensor<T> b_storage = inner_is_row_major(b) ? b : Tensor<T>::contiguous(b);
            const Tensor<T>& a_ref = inner_is_row_major(a) ? a : a_storage;
            const Tensor<T>& b_ref = inner_is_row_major(b) ? b : b_storage;

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

            Tensor<T> result(output_shape, a.device); // we have our result here

            // Create broadcast views for the full shapes (including matrix dims)
            std::vector<int> full_a_shape = batch_shape;
            full_a_shape.push_back(a_rows);
            full_a_shape.push_back(a_cols);

            std::vector<int> full_b_shape = batch_shape;
            full_b_shape.push_back(b_rows);
            full_b_shape.push_back(b_cols);

            Tensor<T> a_view = Tensor<T>::makeBroadcastView(a_ref, full_a_shape);
            Tensor<T> b_view = Tensor<T>::makeBroadcastView(b_ref, full_b_shape);

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

                cuda::launch_gemm_broadcasted<cuda_type_trait_t<T>>(
                    a_ref.data, b_ref.data, result.data,
                    a_rows, a_cols, b_cols,
                    T(1), T(0),
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
                T* mat_a_ptr = a_view.data + offset_a;
                T* mat_b_ptr = b_view.data + offset_b;
                T* result_ptr = result.data + batch_index * matrix_size_result;



                // Use Eigen for the actual matrix multiplication - actual batched mat mul
                Eigen::Map<const MatrixRowMajorT<T>> mat_a(mat_a_ptr, a_rows, a_cols);
                Eigen::Map<const MatrixRowMajorT<T>> mat_b(mat_b_ptr, b_rows, b_cols);
                Eigen::Map<MatrixRowMajorT<T>> result_mat(result_ptr, a_rows, b_cols);

                result_mat =( mat_a * mat_b);
            }

            return result;
        }



        // TODO: write CUDA kernel
        // Maybe I was thinking too much as this works
        template<typename T>
        Tensor<T> reduce(const Tensor<T>& a, std::vector<int>& afterShape, reductions::ReductionOps op){
            Tensor<T> b = a;
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

        template<typename T>
        Tensor<T> hadamard(const Tensor<T> &a, const Tensor<T> &other) {
            if (a.shape != other.shape){
                throw std::invalid_argument("Tensors must have the same shape");
            }

            if (a.device != other.device) {
                throw std::invalid_argument("Tensors must be on the same device");
            }

            // CUDA
            if (a.device == DeviceType::CUDA) {
                Tensor<T> C(a.shape, a.device);

                cuda::launch_elementwise_broadcast<cuda_type_trait_t<T>>(a.data, other.data, C.data, a.getStrides(), other.getStrides(), C.getShape(), OP_Code::OP_MUL);
                return C;
            }

            Tensor<T> result(a.shape, a.device);
            for (ll i = 0; i < a.sizeOfTensor(); i++){
                result.data[i] = a.data[i] * other.data[i];
            }
            return result;
        }


        // -----------------------------------------------Operations helpful in mask generations------------------

        template<typename T>
        Tensor<T> compare(const Tensor<T>& a, const Tensor<T>& b, CompareOp op, T true_val, T false_val) {
            if (a.getShape() != b.getShape()) {
                throw std::runtime_error("In compare and shapes don't match");
            }
            Tensor<T> result(a.getShape(), a.device);

            if (a.device != b.device) {
                throw std::invalid_argument("Tensors must be on the same device");
            }


            if (a.device.type == DeviceType::CUDA) {
                switch (op) {
                    case CompareOp::GT: // greater than
                        bearml::cuda::launch_comparison_kernel<cuda_type_trait_t<T>>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::GE: // greater than or equal to
                        bearml::cuda::launch_comparison_kernel<cuda_type_trait_t<T>>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::LT: // less than
                        bearml::cuda::launch_comparison_kernel<cuda_type_trait_t<T>>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::LE: // less than or equal to
                        bearml::cuda::launch_comparison_kernel<cuda_type_trait_t<T>>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::EQ: // equal to
                        bearml::cuda::launch_comparison_kernel<cuda_type_trait_t<T>>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
                        break;
                    case CompareOp::NE: // not equal to
                        bearml::cuda::launch_comparison_kernel<cuda_type_trait_t<T>>(a.data, b.data, result.data, a.sizeOfTensor(), op, nullptr);
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

        template<typename T>
        Tensor<T> mask_of_greater_than_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val) {
            return compare(first, other, CompareOp::GE, first_val, second_val);
        }

        template<typename T>
        Tensor<T> mask_of_greater_than(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val) {
            return compare(first, other, CompareOp::GT, first_val, second_val);
        }

        template<typename T>
        Tensor<T> mask_of_less_than_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val) {
            return compare(first, other, CompareOp::LE, first_val, second_val);
        }

        template<typename T>
        Tensor<T> mask_of_less_than(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val) {
            return compare(first, other, CompareOp::LT, first_val, second_val);
        }

        template<typename T>
        Tensor<T> mask_of_equal_to(const Tensor<T>& first, const Tensor<T>& other,  T first_val, T second_val){
            return compare(first, other, CompareOp::EQ, first_val, second_val);
        }



        // Double and Tensor
        template<typename T>
        Tensor<T> mask_of_greater_than_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val) {

            Tensor<T> result(other.getShape(), other.device);

            for (size_t i = 0; i < other.sizeOfTensor(); i++) {
                result.data[i] = (first >= other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_greater_than(T first, const Tensor<T>& other,  T first_val, T second_val){
            Tensor<T> result(other.getShape(), other.device);

            for (size_t i = 0; i < other.sizeOfTensor(); i++) {
                result.data[i] = (first > other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_less_than_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val){
            Tensor<T> result(other.getShape(), other.device);

            for (size_t i = 0; i <  other.sizeOfTensor(); i++) {
                result.data[i] = (first <= other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_less_than(T first, const Tensor<T>& other,  T first_val, T second_val){
            Tensor<T> result(other.getShape(), other.device);

            for (size_t i = 0; i <  other.sizeOfTensor(); i++) {
                result.data[i] = (first < other.data[i]) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_equal_to(T first, const Tensor<T>& other,  T first_val, T second_val){
           Tensor<T> result(other.getShape(), other.device);
            for (size_t i = 0; i < other.sizeOfTensor(); ++i) {
                result.data[i] = (std::abs(first - other.data[i]) < 1e-12) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        // Tensor and Double
        template<typename T>
        Tensor<T> mask_of_greater_than_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val){
            Tensor<T> result(first.getShape(), first.device);

            for (size_t i = 0; i < first.sizeOfTensor(); i++) {
                result.data[i] = (first.data[i] >= other) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_greater_than(const Tensor<T>& first, T other,  T first_val, T second_val) {
            Tensor<T> result(first.getShape(), first.device);
            for (size_t i = 0; i < first.sizeOfTensor(); ++i) {
                result.data[i] = (first.data[i] > other) ? first_val: second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_less_than_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val){
            Tensor<T> result(first.getShape(), first.device);

            for (size_t i = 0; i < first.sizeOfTensor(); i++) {
                result.data[i] = (first.data[i] <= other) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_less_than(const Tensor<T>& first, T other,  T first_val, T second_val){
            Tensor<T> result(first.getShape(), first.device);

            for (size_t i = 0; i < first.sizeOfTensor(); i++) {
                result.data[i] = (first.data[i] < other) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        template<typename T>
        Tensor<T> mask_of_equal_to(const Tensor<T>& first, T other,  T first_val, T second_val){
            Tensor<T> result(first.getShape(), first.device);
            for (size_t i = 0; i < first.sizeOfTensor(); ++i) {
                result.data[i] = (std::abs(first.data[i] - other) < 1e-12) ? first_val : second_val;
            }
            // TODO: CUDA implementation

            return result;
        }

        // TODO: templatize
        template<typename T>
        Tensor<T> sign(const Tensor<T>& a){
            Tensor<T> result(a.getShape(), a.device);

            if (a.getDevice().type == DeviceType::CPU) {
                for (size_t i = 0; i < a.sizeOfTensor(); ++i) {
                    result.data[i] = (std::abs(a.data[i] - T(0)) < 1e-12) ? T(0) : ((a.data[i] < T(0)) ? T(-1) : T(1));
                }
                return result;
            }

            // CUDA implementation
            cuda::launch_sign_contiguous<T>(a.data, result.data, a.getShape());

            return result;
        }

        // inverse matrix
        template<typename T>
        Tensor<T> inverse(const Tensor<T>& a) {
            // inverse is only defined for square matrices
            std::vector<int> a_shape = a.getShape();

            if (a_shape.size() < 2) {
                throw std::runtime_error("inverse: only square matrices are supported. Passed in a vector or a scalar.");
            }

            if (a_shape[a_shape.size() - 1] != a_shape[a_shape.size() - 2]) {
                throw std::runtime_error("inverse: only square matrices are supported");
            }

            Tensor<T> result = a; // copy the input tensor

            if (a.getDevice().type == DeviceType::CPU) {

                for (int batch = 0; batch < a.sizeOfTensor() / (a_shape[a_shape.size() - 1] * a_shape[a_shape.size() - 2]); ++batch) {
                    int row = a_shape[a_shape.size() - 1];
                    int col = a_shape[a_shape.size() - 2];
                    Eigen::Map<MatrixRowMajorT<T>> matrix(result.data + batch * row * col, row, col); // get the matrix for this batch
                    Eigen::FullPivLU<MatrixRowMajorT<T>> lu(matrix);
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
        //                 col0   col1   col2   col3
        //               (pos00)(pos01)(pos10)(pos11)
        // row0 (kh=0,kw=0):  1     2     5     6
        // row1 (kh=0,kw=1):  2     3     6     7
        // row2 (kh=0,kw=2):  3     4     7     8
        // row3 (kh=1,kw=0):  5     6     9    10
        // row4 (kh=1,kw=1):  6     7    10    11
        // row5 (kh=1,kw=2):  7     8    11    12
        // row6 (kh=2,kw=0):  9    10    13    14
        // row7 (kh=2,kw=1): 10    11    14    15
        // row8 (kh=2,kw=2): 11    12    15    16

        // TODO: optimize this and implement CUDA version
        template<typename T>
        Tensor<T> im2col_2d(Tensor<T>& a, int kernel_size, int stride, int padding, int dilation){

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

            T padding_value = T(0);

            // patch matrix shape is num_channels*K*K, H_out*W_out

            int num_rows_in_patch = num_channels*kernel_size*kernel_size; // C_in * K * K
            int patch_size = H_out * W_out; // H_out * W_out

            std::vector<int> patch_shape = {batch_size, num_rows_in_patch, patch_size};
            Tensor<T> patch(patch_shape, a.getDevice());

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
                                    int h_in = h * stride - padding + kh*dilation; // h*stride gives the global row index and then -1*padding ensures we start from the correct row by skipping padding rows and kh * dilation tells us -> "Inside the kernel, which row are we on?" Dilation=1 -> just kh. Dilation=2 -> skip a pixel each step.
                                    int w_in = w * stride - padding + kw*dilation;
                                    T value = padding_value;
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
