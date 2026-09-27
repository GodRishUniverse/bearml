// Tensor member definitions - declared in Tensor.h, instantiated at the bottom
#include "Tensor.h"

namespace bearml {

    template <typename T>
    Tensor<T> Tensor<T>::makeBroadcastView(const Tensor &t, const std::vector<int>& newShape){
        Tensor v;            // default-constructed
        v.device = t.device; // same device (broadcasting)
        v.storage = t.storage;
        v.tensor_shape = t.tensor_shape;
        v.tensor_shape.shape   = newShape;
        v.tensor_shape.strides = utils::computeBroadcastStrides(t.tensor_shape.shape, t.tensor_shape.strides, newShape); // no need to change this
        v.tensor_shape.strides_col_major = compute_col_major_strides(newShape); // cached canonical col-major strides for this shape
        return v;
    }

    template <typename T>
    Tensor<T> Tensor<T>::makeSliceView(const Tensor& t, utils::SliceReturn sliceResult){
        Tensor v;
        v.device     = t.device;
        v.storage      = t.storage;
        v.tensor_shape.data_offset = sliceResult.offset;
        v.tensor_shape.shape      = sliceResult.shape;
        v.tensor_shape.strides    = sliceResult.strides;
        v.tensor_shape.strides_col_major = compute_col_major_strides(sliceResult.shape); // cached canonical col-major strides for this shape;
        v.tensor_shape.is_sliced_view = true;
        return v;
    }

    template <typename T>
    Tensor<T> Tensor<T>::makeStrideView(const Tensor& t){
        Tensor v;
        v.device            = t.device;
        v.storage              = t.storage;
        v.tensor_shape = t.tensor_shape;
        return v;
    }

    template <typename T>
    bool Tensor<T>::isScalar(const Tensor& t){
        if (t.tensor_shape.shape.empty()) return true;
        return std::all_of(t.tensor_shape.shape.begin(), t.tensor_shape.shape.end(), [](int dim) { return dim == 1; });
    }

    template <typename T>
    Scalar<T> Tensor<T>::getScalarValue(const Tensor& t){
        if (!isScalar(t)) {
            throw std::runtime_error("Cannot get scalar value from non-scalar tensor");
        }

        // a view's element lives at data_offset, not 0
        T value;
        if (t.device.is_cpu()) {
            value = t.at(t.tensor_shape.data_offset);
        } else {
            // we copy to the host (cpu) to get the scalar value
            t.storage->allocator().copy_to_host(&value, t.const_data() + t.tensor_shape.data_offset, sizeof(T));
        }
        return Scalar<T>(value);
    }

    template <typename T>
    std::vector<int> Tensor<T>::flatten_(int start_dim, int end_dim, bool keepdims){

        // default value of end_dim is -1
        if (end_dim == -1){
            end_dim = this->tensor_shape.shape.size()-1;
        }

        if (start_dim <0 || start_dim>=this->tensor_shape.shape.size() || start_dim>end_dim ||  end_dim>=this->tensor_shape.shape.size() || end_dim  < 0){
            throw std::invalid_argument("Start Dim and End Dims not the appropriate ranges-> CHECK");
        }

        std::vector<int> temp;
        ll total = 1;
        for (ll i = 0; i < this->tensor_shape.shape.size(); i++){
            if (keepdims){
                if (i>=start_dim && i<=end_dim){
                    total *= this->tensor_shape.shape[i];
                    temp.push_back((i ==end_dim) ? total: 1);
                } else{
                    temp.push_back(this->tensor_shape.shape[i]);
                }
            } else{
                if (i<start_dim || i>end_dim){
                    temp.push_back(this->tensor_shape.shape[i]);
                } else {
                    total *= this->tensor_shape.shape[i];
                    if (i == end_dim) temp.push_back(total);
                }
            }

        }

        return temp;
    }

    template <typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor &a, const Tensor &b){
        utils::errorCheckSameDevice(a, b); // will throw an error if devices don't match

        // 5 cases that need to be checked for this
        //                  case 0: A and B are scalars in the form of Tensors [TAKEN INSIDE CASE 1 and 2]
        // case 1: A is scalar
        // case 2: B is scalar
        // case 3: A is a vector and B is a vector
        // case 4: A is matrix and B is a vector
        // case 5: A is a vector and B is a matrix
        // case 6: matrix multiplication - batched and unbatched (GEMM operations)

        // CASE 1
        if (isScalar(a)){
            return elementwise_scalar(b, getScalarValue(a).value(), OP_Code::OP_MUL, LHS_RHS_Code::OP_RHS); // b is not a scalar
        }

        // CASE 2
        if (isScalar(b)){
            return elementwise_scalar(a, getScalarValue(b).value(), OP_Code::OP_MUL, LHS_RHS_Code::OP_RHS); // a is not a scalar
        }

        std::vector<int> a_shape = a.getShape();
        std::vector<int> b_shape = b.getShape();

        // CASE 3: vector-vector product - dot product
        if (a_shape.size() == 1 && b_shape.size()==1){
            if (a_shape[0] != b_shape[0]) {
                throw std::invalid_argument("Vector dimensions must match for dot product");
            }

            if (a.device == DeviceType::CUDA) {
                // Treat as 1x1 matmul: (1,K) * (K,1) = (1,1)
                // 1D row-major vec of length K, viewed as (1,K): row_stride doesn't matter (only one row), col_stride=1
                // 1D row-major vec of length K, viewed as (K,1): row_stride=1, col_stride doesn't matter (only one col)
                Tensor result({1}, a.device);
                cuda::launch_gemm_contiguous<cuda_type_trait_t<T>>(
                    cuda_ptr(a.const_data()), cuda_ptr(b.const_data()), cuda_ptr(result.mutable_data()),
                    1,    // batchsize
                    1,    // m (rows of a treated as row vector)
                    a_shape[0], // k (common dim),
                    1,    // n (cols of result)
                    1.0, 0.0,
                    (int64_t)a_shape[0], 1,   // ra_row, ra_col for A as (1,K)
                    1, 1,                     // rb_row, rb_col for B as (K,1)
                    nullptr
                );
                return result;
            }

            Eigen::Map<const VectorXT<T>> vec_a(a.const_data(), a_shape[0]);
            Eigen::Map<const VectorXT<T>> vec_b(b.const_data(), b_shape[0]);

            Tensor output({1});
            output.at(0) = vec_a.dot(vec_b);
            return output;
        }

        // CASE 4: matrix n by m multiplied with m by 1 vector
        if (a_shape.size() == 2 && b_shape.size() == 1) {
            if (a_shape[1] != b_shape[0]) {
                throw std::invalid_argument("Matrix columns must match vector size");
            }

            if (a.device == DeviceType::CUDA) {
                // Treat vector b as (m,1) matrix: (n,m) * (m,1) = (n,1)
                // A's strides come straight from the tensor (handles transposed/permuted A view natively).
                // B is 1D row-major contiguous → viewed as (m,1): row_stride=1, col_stride=1 (single col).
                Tensor result({a_shape[0]}, a.device);
                cuda::launch_gemm_contiguous<cuda_type_trait_t<T>>(
                    cuda_ptr(a.const_data()), cuda_ptr(b.const_data()), cuda_ptr(result.mutable_data()),
                    1,           // batchsize
                    a_shape[0],  // m
                    a_shape[1],  // k (common dim),
                    1,           // n (output cols)
                    1.0, 0.0,
                    (int64_t)a.tensor_shape.strides[0], (int64_t)a.tensor_shape.strides[1],   // A real strides
                    1, 1,                                            // B as (K,1)
                    nullptr
                );
                return result;
            }

            // Eigen's RowMajor Map can't read a col-major view directly; densify if needed.
            // The CUDA path above already handles layout natively via per-operand strides.
            Tensor a_use = a.is_row_major_contiguous() ? a : Tensor::contiguous(a);
            Eigen::Map<const MatrixRowMajorT<T>> mat_a(a_use.const_data(), a_shape[0], a_shape[1]);
            Eigen::Map<const VectorXT<T>> vec_b(b.const_data(), b_shape[0]);

            Tensor result({a_shape[0]});
            Eigen::Map<VectorXT<T>> result_vec(result.mutable_data(), a_shape[0]);
            result_vec = mat_a * vec_b;

            return result;
        }

        // CASE 5: vector m multiplied with m by n matrix
        if (a_shape.size() == 1 && b_shape.size() == 2) {
            if (a_shape[0] != b_shape[0]) {
                throw std::invalid_argument("Matrix columns must match vector size: : vector m multiplied with m by n matrix");
            }

            if (a.device == DeviceType::CUDA) {
                // Treat vector a as (1,m) matrix: (1,m) * (m,n) = (1,n)
                // A is 1D row-major contiguous → viewed as (1,m): row_stride=m (single row), col_stride=1.
                // B's strides come straight from the tensor (handles transposed/permuted B view).
                Tensor result({1,b_shape[1]}, a.device);
                cuda::launch_gemm_contiguous<cuda_type_trait_t<T>>(
                    cuda_ptr(a.const_data()), cuda_ptr(b.const_data()), cuda_ptr(result.mutable_data()),
                    1,           // batchsize
                    1,           // m
                    b_shape[0],  // k (common dim),
                    b_shape[1],  // n (output cols)
                    1.0, 0.0,
                    (int64_t)a_shape[0], 1,                          // A as (1,K)
                    (int64_t)b.tensor_shape.strides[0], (int64_t)b.tensor_shape.strides[1],    // B real strides
                    nullptr
                );
                return result;
            }

            // For v(m) * B(m,n) -> (n,), the math is B^T * v with v as a column vector.
            // (The earlier `mat_b * vec_a` form only typechecked when B was square.)
            // Densify b first if it's a col-major / strided view so the RowMajor Map is correct.
            Tensor b_use = b.is_row_major_contiguous() ? b : Tensor::contiguous(b);
            Eigen::Map<const VectorXT<T>> vec_a(a.const_data(), a_shape[0]);
            Eigen::Map<const MatrixRowMajorT<T>> mat_b(b_use.const_data(), b_shape[0], b_shape[1]);

            Tensor result({b_shape[1]});
            Eigen::Map<VectorXT<T>> result_vec(result.mutable_data(), b_shape[1]);
            result_vec = mat_b.transpose() * vec_a;

            return result;
        }

        // CASE 6: unbatched matmul
        if (a_shape.size() == 2 && b_shape.size() == 2) {
            if (a_shape[1] != b_shape[0]) {
                throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
            }

            // Operands can be: ROW_MAJOR (fresh tensors, regular slices), COL_MAJOR (transposed
            // views — strides swapped against shape), or STRIDED (permutes, broadcast). The CUDA
            // kernel consumes any (row_stride, col_stride) pair natively, so we pass the operand's
            // own strides for row/col-major. For STRIDED we densify on entry — Eigen Stride<> would
            // also work but adds a lot of code surface for an uncommon case.
            using Layout = utils::Layout;
            auto a_layout = a.layout();
            auto b_layout = b.layout();
            // Tensor::contiguous() is a no-op when the operand is row-major contig, so this
            // costs nothing for the common case.
            Tensor a_use = (a_layout == Layout::STRIDED) ? Tensor::contiguous(a) : a;
            Tensor b_use = (b_layout == Layout::STRIDED) ? Tensor::contiguous(b) : b;
            // refresh layouts after the potential densify
            a_layout = a_use.layout();
            b_layout = b_use.layout();

            if (a.device == DeviceType::CUDA) {
                Tensor result({a_shape[0], b_shape[1]}, a.device);
                // a_use.strides[0] is the M-axis stride, a_use.strides[1] is the K-axis stride.
                // For row-major (M,K) that's (K, 1); for col-major it's (1, M). The kernel doesn't
                // care which — it just uses both to compute a[row * row_stride + k * col_stride].
                cuda::launch_gemm_contiguous<cuda_type_trait_t<T>>(
                    cuda_ptr(a_use.const_data()), cuda_ptr(b_use.const_data()), cuda_ptr(result.mutable_data()),
                    1,           // batchsize
                    a_shape[0],  // m
                    a_shape[1],  // k
                    b_shape[1],  // n
                    1.0, 0.0,
                    (int64_t)a_use.tensor_shape.strides[0], (int64_t)a_use.tensor_shape.strides[1],
                    (int64_t)b_use.tensor_shape.strides[0], (int64_t)b_use.tensor_shape.strides[1],
                    nullptr
                );
                return result;
            }

            // CPU path: Eigen Map can't take a (stride0, stride1) pair, only "is the data laid out
            // RowMajor or ColMajor for THIS shape". So pick the Map type from the layout flag.
            // Eigen happily multiplies a RowMajor matrix by a ColMajor one and vice versa.
            Tensor result({a_shape[0], b_shape[1]});
            Eigen::Map<MatrixRowMajorT<T>> result_mat(result.mutable_data(), a_shape[0], b_shape[1]);

            using ColMajorT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
            if (a_layout == Layout::ROW_MAJOR && b_layout == Layout::ROW_MAJOR) {
                Eigen::Map<const MatrixRowMajorT<T>> mat_a(a_use.const_data(), a_shape[0], a_shape[1]);
                Eigen::Map<const MatrixRowMajorT<T>> mat_b(b_use.const_data(), b_shape[0], b_shape[1]);
                result_mat = mat_a * mat_b;
            } else if (a_layout == Layout::ROW_MAJOR /* && b is COL_MAJOR */) {
                Eigen::Map<const MatrixRowMajorT<T>> mat_a(a_use.const_data(), a_shape[0], a_shape[1]);
                Eigen::Map<const ColMajorT>          mat_b(b_use.const_data(), b_shape[0], b_shape[1]);
                result_mat = mat_a * mat_b;
            } else if (b_layout == Layout::ROW_MAJOR /* && a is COL_MAJOR */) {
                Eigen::Map<const ColMajorT>          mat_a(a_use.const_data(), a_shape[0], a_shape[1]);
                Eigen::Map<const MatrixRowMajorT<T>> mat_b(b_use.const_data(), b_shape[0], b_shape[1]);
                result_mat = mat_a * mat_b;
            } else { // both col-major
                Eigen::Map<const ColMajorT> mat_a(a_use.const_data(), a_shape[0], a_shape[1]);
                Eigen::Map<const ColMajorT> mat_b(b_use.const_data(), b_shape[0], b_shape[1]);
                result_mat = mat_a * mat_b;
            }

            return result;
        }

        // CASE 6: batched matmul (delegates to batchedMatMul which handles CUDA)
        if (a_shape.size() >= 2 && b_shape.size() >= 2) {
            return linear_algebra::batchedMatMul(a, b);
        }

        // SHOULD NEVER REACH HERE
        throw std::invalid_argument("SHOULD NOT REACH HERE - Unsupported tensor shapes for multiplication");
    }

    template <typename T>
    Tensor<T> Tensor<T>::max_impl(const Tensor& t, const Tensor& s){
        if (t.getShape() != s.getShape()){
            throw std::invalid_argument("Shapes dont match for max operation");
        }
        if (t.device == DeviceType::CUDA) {
            Tensor result(t.getShape(), t.device);
            cuda::launch_elementwise_contiguous<cuda_type_trait_t<T>>(
                cuda_ptr(t.const_data()), cuda_ptr(s.const_data()), cuda_ptr(result.mutable_data()),
                t.getShape(), OP_Code::OP_MAX
            );
            return result;
        }
        Tensor  a = t; // copied
        for (size_t i =0; i<t.sizeOfTensor(); i++){
            a.at(i) = std::max({t.at(i), s.at(i)});
        }
        return a;
    }

    template <typename T>
    Tensor<T> Tensor<T>::min_impl(const Tensor& t, const Tensor& s){
        if (t.getShape() != s.getShape()){
            throw std::invalid_argument("Shapes dont match for min operation");
        }
        if (t.device == DeviceType::CUDA) {
            Tensor result(t.getShape(), t.device);
            cuda::launch_elementwise_contiguous<cuda_type_trait_t<T>>(
                cuda_ptr(t.const_data()), cuda_ptr(s.const_data()), cuda_ptr(result.mutable_data()),
                t.getShape(), OP_Code::OP_MIN
            );
            return result;
        }
        Tensor  a = t; // copied
        for (size_t i =0; i<t.sizeOfTensor(); i++){
            a.at(i) = std::min({t.at(i), s.at(i)});
        }
        return a;
    }

    template <typename T>
    Tensor<T> Tensor<T>::accumulate_impl(int dim, reductions::ReductionOps op, bool keepdims){
        if (dim<0 || dim>=tensor_shape.shape.size()){
            throw std::invalid_argument("DIM not in the correct range!");
        }

        // edge cases to consider - when we only have a vector then sum will give a scalar
        if (tensor_shape.shape.size()==1 && dim ==0){
            // no need to check keep dims as if keepdims is false then it will be a scalar anyways
            Tensor new_t({1}, this->device);
            new_t.at(0) = (op == reductions::ReductionOps::PROD) ? T(1)
                                     : (op == reductions::ReductionOps::MAX)  ? std::numeric_limits<T>::lowest()
                                     : (op == reductions::ReductionOps::MIN)  ?  std::numeric_limits<T>::max()
                                     : T(0); // even for argmin/argmax initial value is 0
            // argmin/argmax need the running comparison value seeded to the
            // opposite extreme (the index in new_t.data[0] stays 0 by default)
            T value = (op == reductions::ReductionOps::ARG_MAX) ? std::numeric_limits<T>::lowest()
                    : (op == reductions::ReductionOps::ARG_MIN) ? std::numeric_limits<T>::max()
                    : new_t.at(0); // for argmin/argmax
            for (size_t i =0; i < sizeOfTensor(); i++){
                switch (op) {
                    case reductions::ReductionOps::SUM: case reductions::ReductionOps::MEAN:
                        new_t.at(0) += at(i);
                        break;
                    case reductions::ReductionOps::PROD:
                        new_t.at(0) *= at(i);
                        break;
                    case reductions::ReductionOps::MAX:
                        new_t.at(0) = std::max(new_t.at(0) , at(i));
                        break;
                    case reductions::ReductionOps::MIN:
                        new_t.at(0) = std::min(new_t.at(0) , at(i));
                        break;
                    case reductions::ReductionOps::ARG_MAX:
                        if (at(i) > value) {
                            new_t.at(0) = i;
                            value = at(i);
                        }
                        break;
                    case reductions::ReductionOps::ARG_MIN:
                        if (at(i) < value) {
                            new_t.at(0) = i;
                            value = at(i);
                        }
                        break;
                    default:
                        throw std::runtime_error("Unsupported reduction op");
                }
            }
            // for mean, divide by the number of elements (get scalar value)
            if (op == reductions::ReductionOps::MEAN) {
                new_t.at(0) /= sizeOfTensor();
            }
            new_t.to_(this->device);
            return new_t;
        }

        std::vector<int> newShape = tensor_shape.shape;
        int oldDim = newShape[dim]; // same as dim width
        newShape[dim] = 1; // we will change the shape afterwards

        ll offset_new_shape{1};
        for (int d = dim+1; d <newShape.size(); d++){
            offset_new_shape*=newShape[d];
        }

        ll offset_old{offset_new_shape*oldDim};

        Tensor new_t(newShape, this->device);
        T* flat_data = new_t.mutable_data();

        if (this->device.type == DeviceType::CUDA) {
            cuda::launch_accumulate_kernel<cuda_type_trait_t<T>>(cuda_ptr(this->mutable_data()), cuda_ptr(flat_data), this->tensor_shape.shape, newShape, sizeOfTensor(), offset_new_shape, offset_old, op,  keepdims);

        } else {

            // gpu direct access not allowed, so we copy to CPU first
            Tensor copy_tensor = *this;
            copy_tensor.to_(Device(DeviceType::CPU, -1));

            ll dest_idx = 0;
            for (size_t v =0; v<sizeOfTensor(); v+=offset_old){
                for (size_t s = 0; s<offset_new_shape;s++){
                    // accumulate initial value based on reduction op
                    T val = (op == reductions::ReductionOps::PROD) ? T(1)
                                             : (op == reductions::ReductionOps::MAX || op == reductions::ReductionOps::ARG_MAX)  ? std::numeric_limits<T>::lowest()
                                             : (op == reductions::ReductionOps::MIN || op == reductions::ReductionOps::ARG_MIN)  ?  std::numeric_limits<T>::max()
                                             : T(0);
                    int64_t arg_idx = 0;

                    for (int idx = 0; idx<oldDim; idx++){
                        // edited from this->data to copy_tensor.data
                        T elem = copy_tensor.at(v + idx * offset_new_shape + s);
                        switch (op) {
                            case reductions::ReductionOps::SUM: case reductions::ReductionOps::MEAN:
                                val += elem;
                                break;
                            case reductions::ReductionOps::PROD:
                                val *= elem;
                                break;
                            case reductions::ReductionOps::MAX:
                                val = std::max(val, elem);
                                break;
                            case reductions::ReductionOps::MIN:
                                val = std::min(val, elem);
                                break;
                            case reductions::ReductionOps::ARG_MAX:
                                if (elem > val) {
                                    val = elem;
                                    arg_idx = idx;
                                }
                                break;
                            case reductions::ReductionOps::ARG_MIN:
                                if (elem < val) {
                                    val = elem;
                                    arg_idx = idx;
                                }
                                break;
                            default:
                                throw std::runtime_error("Unsupported reduction op");

                        }
                    }
                    if (op == reductions::ReductionOps::ARG_MAX || op == reductions::ReductionOps::ARG_MIN) {
                        flat_data[dest_idx] = arg_idx;  // store the index, not the value
                    } else {
                        if (op == reductions::ReductionOps::MEAN) val /= oldDim; // only for mean
                        flat_data[dest_idx]= val;
                    }
                    dest_idx++;
                }
            }
        }


        if (keepdims) {
            // new_t.to_(this->device);
            return new_t;
        }
        new_t.flatten_inplace((dim<tensor_shape.shape.size()-1)? dim : (dim-1),  (dim<tensor_shape.shape.size()-1) ? dim+1 : -1, keepdims); // PROBLEM FOUND HERE in case when the dim passed in dim= shape.size()-1
        // new_t.to_(this->device);
        return new_t;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose_view() const{
        // scalar / single-element: nothing to do
        if (sizeOfTensor() <= 1) {
            return makeStrideView(*this);
        }

        if (this->tensor_shape.shape.size() < 2){
            throw std::invalid_argument("transpose requires a tensor of rank >= 2");
        }

        Tensor v = makeStrideView(*this);
        const int n = (int)v.tensor_shape.shape.size();
        std::swap(v.tensor_shape.shape[n-2],             v.tensor_shape.shape[n-1]);
        std::swap(v.tensor_shape.strides[n-2],           v.tensor_shape.strides[n-1]);
        std::swap(v.tensor_shape.strides_col_major[n-2], v.tensor_shape.strides_col_major[n-1]);
        return v;
    }

    template <typename T>
    Tensor<T>::Tensor(std::vector<int> sizePassed, const Device& device) :tensor_shape{sizePassed}, device(device){ // we own the data here
        // shape cannot have 0 or negatives in it
        if (utils::negOrZeroInSizeCheck(this->tensor_shape.shape)){
            throw std::invalid_argument("Size cannot have a negative or zero");
        }

        // check if we even have a cuda capable device
        if (device.type== DeviceType::CUDA){
            int cuda_device_count = 0;
            CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
            if (cuda_device_count == 0) {
                throw std::invalid_argument("No CUDA-capable devices available");
            }
            // check if assigning to wrong CUDA device
            if (device.device_id >= cuda_device_count) {
                   throw std::invalid_argument("Invalid CUDA device ID: " +
                                              std::to_string(device.device_id));
            }
        }
        computeStrides(); // compute strides

        allocate_storage();

        size_t sizeTensor = sizeOfTensor();
        size_t bytes = sizeTensor*sizeof(T);

        // initialize to zero
        if (this->device.is_cpu()) {
            std::fill_n(this->mutable_data(), sizeTensor, T(0));
        } else {
            // Zero initialize on GPU
            CUDA_CHECK(cudaMemset(this->mutable_data(), 0, bytes));
        }

    };

    template <typename T>
    Tensor<T>::Tensor(const Tensor& other) : tensor_shape{other.tensor_shape.shape}, device(other.device){ // we own the data when we copy;
        computeStrides();
        allocate_storage();
        size_t full_size = sizeOfTensor();
        size_t bytes = full_size * sizeof(T);

        if (other.is_contiguous()) {
            // same device: memcpy on cpu, D2D on gpu
            this->storage->allocator().copy_device_to_device(this->mutable_data(), other.const_data() + other.tensor_shape.data_offset, bytes);
        } else {
            // other is a strided view (transpose / permute / slice / broadcast).
            // gather it into our freshly allocated dst, which is row-major for `shape`.
            const int nd = (int)other.tensor_shape.shape.size();
            if (other.device.type == DeviceType::CUDA) {
                // device kernel wants shape/strides on the device; pack them into size_t and hand off
                std::vector<size_t> h_shape(nd);
                std::vector<size_t> h_strides(nd);
                for (int d = 0; d < nd; ++d) {
                    h_shape[d]   = (size_t)other.tensor_shape.shape[d];
                    h_strides[d] = (size_t)other.tensor_shape.strides[d];
                }
                cuda::utils::launch_contiguous_gather<cuda_type_trait_t<T>>(
                    cuda_ptr(other.const_data()), cuda_ptr(this->mutable_data()), other.tensor_shape.data_offset,
                    h_shape.data(), h_strides.data(), (size_t)nd, full_size);
            } else {
                // i walks the DESTINATION in row-major (dst is dense, so we write data[i] in flat order)
                for (size_t i = 0; i < full_size; ++i) {
                    size_t tmp = i;                 // running quotient — peels one axis at a time
                    size_t src = other.tensor_shape.data_offset; // start at the view's offset into the source's storage
                    // innermost dim varies fastest in row-major, so we modulo it out first and work outwards
                    for (int d = nd - 1; d >= 0; --d) {
                        size_t coord = tmp % (size_t)other.tensor_shape.shape[d];   // coord along axis d for this dst index
                        tmp /= (size_t)other.tensor_shape.shape[d];                 // strip that axis from tmp for the next iteration
                        src += coord * (size_t)other.tensor_shape.strides[d];       // step in source storage by source's own stride along axis d
                    }
                    this->at(i) = other.at(src);
                }
            }
        }
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor& other){
        if (this != &other) {
            Tensor copy(other);        // copy ctor densifies views
            *this = std::move(copy);
        }
        return *this;
    }

    template <typename T>
    Tensor<T>::Tensor(Tensor&& other) noexcept : tensor_shape(std::move(other.tensor_shape)), storage(std::move(other.storage)), device(other.device), graph_node(std::move(other.graph_node)){
        other.tensor_shape.is_sliced_view = false;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept{
        if (this != &other) {
            this->tensor_shape = std::move(other.tensor_shape);
            this->storage = std::move(other.storage);
            this->device = other.device;
            this->graph_node = std::move(other.graph_node);
            other.tensor_shape.is_sliced_view = false;
        }
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::to(const Device& targetDevice) const{
        if (device == targetDevice) {
            return *this; // Return copy on same device
        }
        // views are densified first so bytes match the shape
        bool exact_buffer = is_contiguous() && tensor_shape.data_offset == 0 && storage_elements() == sizeOfTensor();
        Tensor result = exact_buffer ? makeStrideView(*this) : Tensor(*this);
        result.storage = result.storage->to(targetDevice);
        result.device = targetDevice;
        return result;
    }

    template <typename T>
    void Tensor<T>::to_(const Device& targetDevice){
        if (device == targetDevice) return;
        check_inplace();
        auto node = graph_node;
        *this = to(targetDevice);
        graph_node = node; // a moved parameter stays tracked
    }

    template <typename T>
    bool Tensor<T>::is_contiguous() const{
          size_t s = 1;
          for (int d = (int)tensor_shape.shape.size() - 1; d >= 0; --d) {
              if ((size_t)tensor_shape.strides[d] != s) return false;
              s *= tensor_shape.shape[d];
          }
          return tensor_shape.data_offset == 0;
    }

    template <typename T>
    bool Tensor<T>::is_col_major_contiguous() const{
        if (tensor_shape.strides.size() != tensor_shape.shape.size()) return false;
        size_t v = 1;
        for (size_t d = 0; d < tensor_shape.shape.size(); ++d) {
            if ((size_t)tensor_shape.strides[d] != v) return false;
            v *= tensor_shape.shape[d];
        }
        return tensor_shape.data_offset == 0;
    }

    template <typename T>
    utils::Layout Tensor<T>::layout() const{
        if (is_row_major_contiguous()) return utils::Layout::ROW_MAJOR;
        if (is_col_major_contiguous()) return utils::Layout::COL_MAJOR;
        return utils::Layout::STRIDED;
    }

    template <typename T>
    Scalar<T> Tensor<T>::get(std::vector<int> index) const{
        if (!this->device.is_cpu()) {
            throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use get()");
        }
        if (index.size() != tensor_shape.shape.size()){
            throw std::invalid_argument("Invalid index size: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" +utils::debugShapes(this->tensor_shape.shape)+"\n");
        }

        if (utils::isIndexValid(index, this->tensor_shape.shape) == false){
            throw std::invalid_argument("Invalid index shape: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" + utils::debugShapes(this->tensor_shape.shape)+"\n");
        }

        size_t off = 0;
        for (size_t d = 0; d < tensor_shape.shape.size(); ++d)
            off += index[d] * this->tensor_shape.strides[d];
        return at(this->tensor_shape.data_offset+ off);
    }

    template <typename T>
    Scalar<T> Tensor<T>::get(std::span<int> index) const{
        if (!this->device.is_cpu()) {
            throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use get()");
        }
        if (index.size() != tensor_shape.shape.size()){
            throw std::invalid_argument("Invalid index size: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" +utils::debugShapes(this->tensor_shape.shape)+"\n");
        }

        if (utils::isIndexValid(index, this->tensor_shape.shape) == false){
            throw std::invalid_argument("Invalid index shape: \nPassed:\t" + utils::debugShapes(index)+"\nExpected:\t" + utils::debugShapes(this->tensor_shape.shape)+"\n");
        }

        size_t off = 0;
        for (size_t d = 0; d < tensor_shape.shape.size(); ++d)
            off += index[d] * this->tensor_shape.strides[d];
        return at(this->tensor_shape.data_offset+ off);
    }

    template <typename T>
    void Tensor<T>::set_using_bulk_copy(const std::vector<T>& values){
        check_inplace();
        size_t full_size = this->sizeOfTensor();
        // partial fills are allowed; only reject overflow
        if (values.size() > full_size) {
            throw std::invalid_argument("Invalid values size: \nPassed:\t" + std::to_string(values.size()) + "\nExpected (max):\t" + std::to_string(full_size) + "\n");
        }

        if (!this->is_contiguous()) {
            throw std::runtime_error("set_using_bulk_copy only supports contiguous tensors");
        }
        // Makes it device-agnostic: destination first, then source, then bytes
        this->storage->allocator().copy_to_device(this->mutable_data() + this->tensor_shape.data_offset, values.data(), values.size() * sizeof(T));
    }

    template <typename T>
    void Tensor<T>::set(Scalar<T> val, std::vector<int> index){
        check_inplace();

        if (!this->device.is_cpu()) {
            throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use set()");
        }

        if (index.size() != tensor_shape.shape.size()){
            std::string err = "Expected size = ";
            std::string expected_size {"("};
            std::string size_passed{"("};
            for (size_t s = 0; s<tensor_shape.shape.size(); s++){
                expected_size+= std::to_string(tensor_shape.shape[s]);
                if (s-1 != tensor_shape.shape.size()-1){
                    expected_size+= ", ";
                }
            }
            expected_size+=")";
            err+=expected_size;
            err+= " Actual passed in = ";
            for (size_t s = 0; s<index.size(); s++){
                size_passed+= std::to_string(index[s]);
                if (s-1 != index.size()-1){
                    size_passed+= ", ";
                }
            }
            size_passed+=")";
            err+=size_passed;
            err = "Invalid index size: " + err;
            throw std::invalid_argument(err);
        }


        // check negatives and size on each index
        bool flag = false;

        std::string err = "Expected size = ";
        std::string expected_size {"("};
        std::string size_passed{"("};
        for (size_t s = 0; s<tensor_shape.shape.size(); s++){
            expected_size+= std::to_string(tensor_shape.shape[s]);
            if (s-1 != tensor_shape.shape.size()-1){
                expected_size+= ", ";
            }

            size_passed+= std::to_string(index[s]);
            if (s-1 != index.size()-1){
                size_passed+= ", ";
            }

            if (index[s] > tensor_shape.shape[s] || index[s]<0){
                flag = true;
            }
        }
        expected_size+=")";

        err+=expected_size;

        err+= " Actual passed in = ";
        size_passed+=")";

        err+=size_passed;

        err= "Invalid index shape: " + err;

        if (flag){
            throw std::invalid_argument(err);
        }

        size_t off = 0;
        for (size_t d = 0; d < tensor_shape.shape.size(); ++d)
            off += index[d] * tensor_shape.strides[d];

        at(this->tensor_shape.data_offset+ off) = val;
    }

    template <typename T>
    void Tensor<T>::set_with_offset(ll offset, int row, int col, Scalar<T> val){
        check_inplace();
        // assumes offset is passed correctly at the moment
        if (!this->device.is_cpu()) {
            throw std::runtime_error("GPU Direct Memory Access not setup right now! Transfer to cpu to use set()");
        }

        at(offset+row*(this->tensor_shape.shape[this->tensor_shape.shape.size()-1])+col) = val;
    }

    template <typename T>
    void Tensor<T>::printShape() const{
        std::cout << "Shape: [";
        for (ll i = 0; i < this->tensor_shape.shape.size(); i++){
            std::cout << this->tensor_shape.shape[i];
            if (i != this->tensor_shape.shape.size()-1){
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    template <typename T>
    std::vector<int> Tensor<T>::compute_row_major_strides(const std::vector<int>& sh){
        std::vector<int> s(sh.size());
        size_t v = 1;
        for (int d = (int)sh.size() - 1; d >= 0; --d) {
            s[d] = (int)v;
            v *= sh[d];
        }
        return s;
    }

    template <typename T>
    std::vector<int> Tensor<T>::compute_col_major_strides(const std::vector<int>& sh){
        std::vector<int> s(sh.size());
        size_t v = 1;
        for (size_t d = 0; d < sh.size(); ++d) {
            s[d] = (int)v;
            v *= sh[d];
        }
        return s;
    }

    template <typename T>
    void Tensor<T>::computeStrides(){
        tensor_shape.strides            = compute_row_major_strides(tensor_shape.shape);
        tensor_shape.strides_col_major  = compute_col_major_strides(tensor_shape.shape);
    }

    template <typename T>
    Tensor<T> Tensor<T>::ones(const std::vector<int>& shape, const Device& device){
        Tensor t(shape, device);
        t.fill(1.0); // use fill for better performance
        return t;
    }

    template <typename T>
    Tensor<T> Tensor<T>::ones_like(const Tensor& t){
        Tensor result(t.tensor_shape.shape, t.device);
        result.fill(1.0); // use fill for better performance
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::from_host(const std::vector<int>& shape, const std::vector<T>& host, const Device& device){
        Tensor t(shape, device);
        if (host.size() != t.sizeOfTensor()) {
            throw std::invalid_argument("from_host: data size does not match shape: \nPassed:\t" +
                std::to_string(host.size()) + "\nExpected:\t" + std::to_string(t.sizeOfTensor()) + "\n");
        }
        t.set_using_bulk_copy(host);
        return t;
    }

    template <typename T>
    void Tensor<T>::print_recursive(std::ostream& os, const Tensor& t, size_t dim, size_t offset, int indent){
          if (dim == t.tensor_shape.shape.size() - 1) {
              os << "[";
              for (size_t i = 0; i < t.tensor_shape.shape[dim]; ++i) {
                  os << std::setw(9) << std::setprecision(print_precision)
                     << t.at(offset + i * t.tensor_shape.strides[dim]);
                  if (i + 1 < t.tensor_shape.shape[dim]) os << ", ";
              }
              os << "]";
              return;
          }
          os << "[";
          for (size_t i = 0; i < t.tensor_shape.shape[dim]; ++i) {
              if (i > 0) {
                  os << ",\n";
                  // one blank line between 2-D "chunks" when there are >2 outer dims
                  if (dim < t.tensor_shape.shape.size() - 2) os << "\n";
                  os << std::string(indent + 1, ' ');
              }
              print_recursive(os, t, dim + 1, offset + i * t.tensor_shape.strides[dim], indent + 1);
          }
          os << "]";
      }

    template <typename T>
    std::ostream& Tensor<T>::print_impl(std::ostream& os, const Tensor& tensor){


        os << "Tensor with shape: [";
        for (size_t i = 0; i < tensor.tensor_shape.shape.size(); ++i) {
            os << tensor.tensor_shape.shape[i];
            if (i != tensor.tensor_shape.shape.size() - 1) os << ", ";
        }
        os << "]\n";

        os << "Tensor on device: ";
        os << tensor.device.to_string() << "\n";

        os << "Tensor dtype: " << dtype_name(dtype_of<T>) << "\n";

        os << "Tensor data:\n";
        if (tensor.device.is_cpu()) {
            // No copy: walk the view in place.
            if (tensor.tensor_shape.shape.empty()) {
                os << tensor.at(tensor.tensor_shape.data_offset);
            } else {
                print_recursive(os, tensor, 0, tensor.tensor_shape.data_offset, 0);
            }
        } else {
            // copy constructor used here
            Tensor host = tensor.to(Device::cpu());
            if (host.tensor_shape.shape.empty()) {
                os << host.at(host.tensor_shape.data_offset);
            } else {
                print_recursive(os, host, 0, host.tensor_shape.data_offset, 0);
            }
        }
        os << "\n";

        os << "STORAGE REFS: " << tensor.storage.use_count() << std::endl;
        os << "SLICED VIEW: " << ((tensor.getDataOffset() != 0) ? "TRUE" : "FALSE") << std::endl;

        return os;
    }

    template <typename T>
    Tensor<T> Tensor<T>::elementwise_binary(const Tensor& A, const Tensor& B, OP_Code op){
        utils::errorCheckSameDevice(A, B);
        if (A.device.type == DeviceType::CUDA) {
            if (A.tensor_shape.shape == B.tensor_shape.shape) {
                Tensor C(A.tensor_shape.shape, A.device);
                cuda::launch_elementwise_contiguous<cuda_type_trait_t<T>>(cuda_ptr(A.const_data()), cuda_ptr(B.const_data()), cuda_ptr(C.mutable_data()), C.getShape(), op);
                return C;
            }
            auto outShape = utils::computeBroadcastShape(A.tensor_shape.shape, B.tensor_shape.shape);

            Tensor aView = makeBroadcastView(A, outShape);
            Tensor bView = makeBroadcastView(B, outShape);

            // aView.to_(A.device);
            // bView.to_(A.device);

            Tensor C(outShape, A.device);

            cuda::launch_elementwise_broadcast<cuda_type_trait_t<T>>(cuda_ptr(aView.mutable_data()), cuda_ptr(bView.mutable_data()), cuda_ptr(C.mutable_data()),
                aView.getStrides(), bView.getStrides(), C.getShape(), op);
            return C;
        }
        // CPU path with appropriate lambda
        switch(op) {
            case OP_Code::OP_ADD: return elementwise_binary_cpu(A, B, std::plus<T>{});
            case OP_Code::OP_SUB: return elementwise_binary_cpu(A, B, std::minus<T>{});
            case OP_Code::OP_MUL: return elementwise_binary_cpu(A, B, std::multiplies<T>{});
            case OP_Code::OP_DIV: return elementwise_binary_cpu(A, B, std::divides<T>{});
            case OP_Code::OP_MAX: return elementwise_binary_cpu(A, B, [](T a, T b){ return std::max(a,b); });
            case OP_Code::OP_MIN: return elementwise_binary_cpu(A, B, [](T a, T b){ return std::min(a,b); });
            default:
                throw std::invalid_argument("OP Code does not exist - in elementwise_binary.");
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::elementwise_scalar(const Tensor& A, T b, OP_Code op, LHS_RHS_Code side){
        Tensor C(A.tensor_shape.shape, A.device);
        if (A.device.type == DeviceType::CUDA) {
            cuda::launch_elementwise_contiguous_with_constant<cuda_type_trait_t<T>>(cuda_ptr(A.const_data()), cuda_val(b), cuda_ptr(C.mutable_data()), A.tensor_shape.shape, op, side);
            return C;
        }
        size_t N = A.sizeOfTensor();
        if (side == LHS_RHS_Code::OP_RHS) {
            // A op b
            switch(op) {
                case OP_Code::OP_ADD:
                    for (size_t i=0;i<N;++i) C.at(i)=A.at(i)+b;
                    break;
                case OP_Code::OP_SUB:
                    for (size_t i=0;i<N;++i) C.at(i)=A.at(i)-b;
                    break;
                case OP_Code::OP_MUL:
                    for (size_t i=0;i<N;++i) C.at(i)=A.at(i)*b;
                    break;
                case OP_Code::OP_DIV:
                    for (size_t i=0;i<N;++i) C.at(i)=A.at(i)/b;
                    break;
                default:
                    throw std::invalid_argument("OP Code not supported for scalar op.");
            }
        } else {
            // b op A
            switch(op) {
                case OP_Code::OP_ADD:
                    for (size_t i=0;i<N;++i) C.at(i)=b+A.at(i);
                    break;
                case OP_Code::OP_SUB:
                    for (size_t i=0;i<N;++i) C.at(i)=b-A.at(i);
                    break;
                case OP_Code::OP_MUL:
                    for (size_t i=0;i<N;++i) C.at(i)=b*A.at(i);
                    break;
                case OP_Code::OP_DIV:
                    for (size_t i=0;i<N;++i) C.at(i)=b/A.at(i);
                    break;
                default:
                    throw std::invalid_argument("OP Code not supported for scalar op.");
            }
        }
        return C;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::inplace_tensor_binary(const Tensor& other, OP_Code op){
        utils::errorCheckSameDevice(*this, other);
        if (tensor_shape.shape != other.tensor_shape.shape) {
            auto outShape = utils::computeBroadcastShape(tensor_shape.shape, other.tensor_shape.shape);
            // we require that *this already has exactly outShape:
            // otherwise you'd need to reallocate or error.
            if (tensor_shape.shape != outShape)
                throw std::invalid_argument("LHS must match broadcasted shape for in-place op");

            // CUDA
            if (device.type == DeviceType::CUDA) {

                Tensor oView = makeBroadcastView(other, outShape);

                cuda::launch_elementwise_broadcast<cuda_type_trait_t<T>>(cuda_ptr(mutable_data()), cuda_ptr(other.const_data()), cuda_ptr(mutable_data()),
                    getStrides(), oView.getStrides(), outShape, op);

                return *this;
            }

            // CPU broadcast in-place
            Tensor oView = makeBroadcastView(other, outShape);
            size_t N = sizeOfTensor();
            for (size_t idx = 0; idx < N; ++idx) {
                ll tmp = idx, offO = 0;
                for (int d = static_cast<int>(outShape.size()) - 1; d >= 0; --d) {
                    int coord = tmp % outShape[d];
                    tmp /= outShape[d];
                    offO += coord * oView.tensor_shape.strides[d];
                }
                switch(op) {
                    case OP_Code::OP_ADD:
                        at(idx) += oView.at(offO);
                        break;
                    case OP_Code::OP_SUB:
                        at(idx) -= oView.at(offO);
                        break;
                    default:
                        throw std::invalid_argument("Unsupported in-place broadcast op");
                }
            }
        } else {
            // CUDA
            if (device.type == DeviceType::CUDA) {
                cuda::launch_elementwise_contiguous<cuda_type_trait_t<T>>(cuda_ptr(mutable_data()), cuda_ptr(other.const_data()), cuda_ptr(mutable_data()), tensor_shape.shape, op);
                return *this;
            }
            size_t N = sizeOfTensor();
            switch(op) {
                case OP_Code::OP_ADD:
                    for (size_t i=0;i<N;++i) at(i)+=other.at(i);
                    break;
                case OP_Code::OP_SUB:
                    for (size_t i=0;i<N;++i) at(i)-=other.at(i);
                    break;
                default:
                    throw std::invalid_argument("Unsupported in-place contiguous op");
            }
        }
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::inplace_scalar(T b, OP_Code op){
        if (device.type == DeviceType::CUDA) {
            cuda::launch_elementwise_contiguous_with_constant<cuda_type_trait_t<T>>(cuda_ptr(this->mutable_data()), cuda_val(b), cuda_ptr(this->mutable_data()), tensor_shape.shape, op, LHS_RHS_Code::OP_RHS);
            return *this;
        }
        size_t N = sizeOfTensor();
        switch(op) {
            case OP_Code::OP_ADD:
                for (size_t i=0;i<N;++i) at(i)+=b;
                break;
            case OP_Code::OP_SUB:
                for (size_t i=0;i<N;++i) at(i)-=b;
                break;
            default:
                throw std::invalid_argument("Unsupported in-place scalar op");
        }
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::elementwise_unary(const Tensor& A, OP_Code op){
        if (A.device.type == DeviceType::CUDA) {
            Tensor C(A.tensor_shape.shape, A.device);
            cuda::launch_elementwise_unary<cuda_type_trait_t<T>>(
                cuda_ptr(A.const_data()),
                cuda_ptr(C.mutable_data()),
                C.getShape(),
                op
            );
            return C;
        }
        // CPU path with appropriate lambda
        switch(op) {
            case OP_Code::OP_EXP: return elementwise_unary_cpu(A, [](T a){ return std::exp(a); });
            case OP_Code::OP_LOG: return elementwise_unary_cpu(A, [](T a){ return std::log(a); });
            case OP_Code::OP_SQRT: return elementwise_unary_cpu(A, [](T a){ return std::sqrt(a); });
            case OP_Code::OP_ABS: return elementwise_unary_cpu(A, [](T a){ return std::abs(a); });
            case OP_Code::OP_SIN: return elementwise_unary_cpu(A, [](T a){ return std::sin(a); });
            case OP_Code::OP_COS: return elementwise_unary_cpu(A, [](T a){ return std::cos(a); });
            case OP_Code::OP_TAN: return elementwise_unary_cpu(A, [](T a){ return std::tan(a); });
            case OP_Code::OP_SINH: return elementwise_unary_cpu(A, [](T a){ return std::sinh(a); });
            case OP_Code::OP_COSH: return elementwise_unary_cpu(A, [](T a){ return std::cosh(a); });
            case OP_Code::OP_TANH: return elementwise_unary_cpu(A, [](T a){ return std::tanh(a); });
            default:
                throw std::invalid_argument("OP Code does not exist - in elementwise_unary.");
        }
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor &other){
        check_inplace(&other);
        return inplace_tensor_binary(other, OP_Code::OP_ADD);
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(Scalar<T> b){
        check_inplace();
        return inplace_scalar(b.value(), OP_Code::OP_ADD);
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor &other){
        check_inplace(&other);
        return inplace_tensor_binary(other, OP_Code::OP_SUB);
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(Scalar<T> b){
        check_inplace();
        return inplace_scalar(b.value(), OP_Code::OP_SUB);
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor& B){
        check_inplace(&B);
         auto node = graph_node;
         *this = matmul(*this, B);
         graph_node = node; // in-place ops don't record
         return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(Scalar<T> B){
        check_inplace();
         auto node = graph_node;
         *this = elementwise_scalar(*this, B.value(), OP_Code::OP_MUL, LHS_RHS_Code::OP_RHS);
         graph_node = node; // in-place ops don't record
         return *this;
    }

    template <typename T>
    bool Tensor<T>::equal_impl(const Tensor &a, const Tensor &b){
        if (a.getShape() == b.getShape() && a.getStrides() == b.getStrides()){
            if (a.getDevice().type == DeviceType::CUDA && b.getDevice().type == DeviceType::CUDA){
                return cuda::launch_check_equal_kernel<cuda_type_trait_t<T>>(cuda_ptr(a.const_data()), cuda_ptr(b.const_data()), a.sizeOfTensor()); // need to test this
            }
            // NOTE: std::abs is better for doubles
            for (size_t i = 0; i<a.sizeOfTensor(); i++){
                if ((std::abs(a.at(i)-b.at(i))) >= Tensor::MIN_DIFF){ // check if the error is greater than 10^-15
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    template <typename T>
    Tensor<T> Tensor<T>::accumulate(int dim, reductions::ReductionOps op, bool keepdims){
        Tensor out = accumulate_impl(dim, op, keepdims);
        if (op == reductions::ReductionOps::SUM && keepdims) {
            out.record_op(OP_Code::OP_SUM, {this}, OpAttributes{.dim = dim}, out);
        } else if (graph_node) {
            throw std::logic_error("accumulate: only SUM with keepdims is differentiable");
        }
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::exp(Tensor& t){
        // std::cout <<"EXPONENTIATED" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_EXP);
        out.record_op(OP_Code::OP_EXP, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::sin(Tensor& t){
        // std::cout <<"SIN" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_SIN);
        out.record_op(OP_Code::OP_SIN, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::cos(Tensor& t){
        // std::cout <<"COS" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_COS);
        out.record_op(OP_Code::OP_COS, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::tan(Tensor& t){
        // std::cout <<"TAN" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_TAN);
        out.record_op(OP_Code::OP_TAN, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::sinh(Tensor& t){
        // std::cout <<"SINH" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_SINH);
        out.record_op(OP_Code::OP_SINH, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::cosh(Tensor& t){
        // std::cout <<"COSH" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_COSH);
        out.record_op(OP_Code::OP_COSH, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::tanh(Tensor& t){
        // std::cout <<"TANH" <<std::endl;
        Tensor out = elementwise_unary(t, OP_Code::OP_TANH);
        out.record_op(OP_Code::OP_TANH, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::max(const Tensor& t, Scalar<T> val){
        if (t.graph_node) {
            Tensor constant(t.getShape(), t.device);
            constant.fill(val);
            return max(t, constant);
        }
        // std::cout <<"MAX" <<std::endl;
        if (t.device == DeviceType::CUDA) {
            // Create a scalar tensor on CUDA filled with val
            Tensor scalar_t(t.getShape());
            for (size_t i = 0; i < t.sizeOfTensor(); i++) scalar_t.at(i) = val.value();
            scalar_t.to_(t.device);
            return Tensor::max_impl(t, scalar_t);
        }
        Tensor  a = t; // copied
        for (size_t i =0; i<t.sizeOfTensor(); i++){
            a.at(i) = std::max({t.at(i), val.value()});
        }
        return a;
    }

    template <typename T>
    Tensor<T> Tensor<T>::max(const Tensor& t, const Tensor& s){
        Tensor out = max_impl(t, s);
        out.record_op(OP_Code::OP_MAX, {&t, &s}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::min(const Tensor& t, Scalar<T> val){
        if (t.graph_node) {
            Tensor constant(t.getShape(), t.device);
            constant.fill(val);
            return min(t, constant);
        }
        // std::cout <<"MIN" <<std::endl;
        if (t.device == DeviceType::CUDA) {
            Tensor scalar_t(t.getShape());
            for (size_t i = 0; i < t.sizeOfTensor(); i++) scalar_t.at(i) = val.value();
            scalar_t.to_(t.device);
            return Tensor::min_impl(t, scalar_t);
        }
        Tensor  a = t; // copied
        for (size_t i =0; i<t.sizeOfTensor(); i++){
            a.at(i) = std::min({t.at(i), val.value()});
        }
        return a;
    }

    template <typename T>
    Tensor<T> Tensor<T>::min(const Tensor& t, const Tensor& s){
        Tensor out = min_impl(t, s);
        out.record_op(OP_Code::OP_MIN, {&t, &s}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::abs(const Tensor &t){
        Tensor out = elementwise_unary(t, OP_Code::OP_ABS);
        out.record_op(OP_Code::OP_ABS, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::sqrt(const Tensor &t){
        Tensor out = elementwise_unary(t, OP_Code::OP_SQRT);
        out.record_op(OP_Code::OP_SQRT, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::mean(Tensor &t){
        // sum_kernel accumulates into a buffer of the tensor's own element
        // type, so restrict to floating types (reject int tensors) at compile time.
        // runtime check - Tensor.cpp instantiates every member for int as well
        if constexpr (!bearml::is_floating(bearml::dtype_of<T>)) {
            throw std::logic_error("Tensor::mean supports only float, double, or bfloat16 element types");
        }

        Tensor result({1}, t.device);
        if (t.device == DeviceType::CUDA) {
            // Reduce straight into result's own T buffer; the kernel is templated
            // on the element type, so no separate accumulator tensor is needed.
            // TODO: fix the type
            cuda::launch_sum_kernel(cuda_ptr(t.mutable_data()), cuda_ptr(result.mutable_data()), t.sizeOfTensor());
            result.to_(Device::cpu()); // GPU direct memory access is NOT SET UP
            result.set(static_cast<T>(static_cast<double>(result.at(0)) / static_cast<double>(t.sizeOfTensor())), {0}); // set the mean value
            result.to_(t.device); // send back to the original device
        }else {
            size_t sizeTensor = t.sizeOfTensor();
            double sum = 0.0; // accumulate in double on CPU to avoid float cancellation
            for (size_t i =0; i<sizeTensor; i++){
               sum += static_cast<double>(t.at(i));

            }
            result.set(static_cast<T>(sum / static_cast<double>(t.sizeOfTensor())), {0}) ;
        }
        result.record_op(OP_Code::OP_MEAN_FOR_GRAD, {&t}, {}, result);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::log(Tensor &t){
        Tensor out = elementwise_unary(t, OP_Code::OP_LOG);
        out.record_op(OP_Code::OP_LOG, {&t}, {}, out);
        return out;
    }

    template <typename T>
    Tensor<T> Tensor<T>::softmax(Tensor &t, int64_t dim){
        // default dim is -1, which means softmax over the last dimension
        if (dim < -1) {
            throw std::invalid_argument("dim must be >= -1");
        }
        if (dim >= (int64_t)t.tensor_shape.shape.size()) {
            throw std::invalid_argument("dim must be < number of dimensions");
        }
        int dim_to_use = (dim == -1) ? t.tensor_shape.shape.size() - 1 : dim;

        Tensor result(t.tensor_shape.shape, t.device);
        if (t.device.is_cuda()) {
            // CUDA softmax kernels are instantiated for float and double only
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                cuda::launch_softmax_kernel<cuda_type_trait_t<T>>(
                    cuda_ptr(t.mutable_data()),
                    cuda_ptr(result.mutable_data()),
                    t.getShape(),
                    t.getStrides(),
                    t.sizeOfTensor(),
                    dim_to_use
                );
            } else {
                throw std::logic_error("Tensor::softmax on CUDA supports only float and double");
            }
        } else {
            // CPU softmax
            size_t size = t.sizeOfTensor();

            const auto& shape  = t.getShape();
            const auto& stride = t.getStrides();

            size_t num_groups = size / shape[dim_to_use];
            int stride_dim = stride[dim_to_use];
            std::vector<T> sum_values(num_groups, T(0));
            std::vector<T> max_values(num_groups, -std::numeric_limits<T>::infinity());

            // The max trick is applied here to avoid numerical overflow
            for (size_t i = 0; i < num_groups; ++i) {

                size_t row_base = 0 , grp = i;
                for (int d = (int)shape.size() - 1; d >= 0; --d) {
                    if (d == dim_to_use) continue;
                    row_base += (grp % shape[d]) * stride[d];
                    grp /= shape[d];
                }

                for (size_t j = 0; j < shape[dim_to_use]; ++j) {
                    size_t index = row_base + j * stride_dim;
                    max_values[i] = std::max(max_values[i], t.at(index));
                }

                for (size_t j = 0; j < shape[dim_to_use]; ++j) {
                    size_t index = row_base + j * stride_dim;
                    result.at(index) = std::exp(t.at(index)-max_values[i]);
                    sum_values[i] += result.at(index);
                }

                for (size_t j = 0; j < shape[dim_to_use]; ++j) {
                    size_t index = row_base + j * stride_dim;
                    result.at(index)/=sum_values[i];
                }
            }

        }

        result.record_op(OP_Code::OP_SOFTMAX, {&t}, OpAttributes{.dim = dim_to_use}, result);
        return result;
    }

    template <typename T>
    void Tensor<T>::fill(Scalar<T> v){
        check_inplace();
        // CUDA fill
        if (!this->device.is_cpu()) {
            cuda::launch_fill<cuda_type_trait_t<T>>(
                cuda_ptr(this->mutable_data()),
                cuda_val(v.value()),
                this->tensor_shape.shape
            );
            return;
        }

        // CPU fill
        for (size_t i =0;i<sizeOfTensor(); i++){
            this->at(i) = v;
        }
    }

    template <typename T>
    bool Tensor<T>::has_nonzero_gradient(Tensor& t){

        if (t.device.is_cpu()) {
            // we only have to check if there is at least one of the numbers that is non-zero
            for (size_t i =0;i<t.sizeOfTensor(); i++){
                if (std::abs(t.at(i)) > 1e-12) {
                    return true;
                }
            }
            return false;
        }

        //CUDA support
        // TODO: fix the type
        return cuda::launch_check_zero_kernel(cuda_ptr(t.const_data()), t.sizeOfTensor());

    }

    template <typename T>
    Tensor<T> Tensor<T>::flatten(int start_dim, int end_dim, bool keepdims){
        Tensor result = *this; // copy the tensor
        std::vector<int> newShape = this->flatten_(start_dim, end_dim, keepdims);
        result.setShape(newShape);
        result.computeStrides();
        return result;
    }

    template <typename T>
    void Tensor<T>::flatten_inplace(int start_dim, int end_dim, bool keepdims){
        check_inplace();
        this->tensor_shape.shape = Tensor::flatten_(start_dim, end_dim, keepdims);
        computeStrides();
    }

    template <typename T>
    Tensor<T>& Tensor<T>::linspace(Scalar<T> start, Scalar<T> end){
        check_inplace();
        size_t long_size = this->sizeOfTensor();
        double size = static_cast<double>(long_size-1);
        // accumulate in double for precision, store back as the element type T
        double cur = static_cast<double>(start);
        double step = (static_cast<double>(end)-cur)/(size);
        if (this->device.is_cpu()) {
            for (size_t i =0; i < long_size ; i++){
                this->at(i) = static_cast<T>(cur);
                cur+=step;
            }
        } else {
            cuda::launch_linspace_kernel<cuda_type_trait_t<T>>(cuda_ptr(this->mutable_data()), cuda_val(static_cast<T>(start)), cuda_val(static_cast<T>(step)), long_size);
        }
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(){
        Tensor v = transpose_view();
        v.record_op(OP_Code::OP_TRANSPOSE, {this}, {}, v);
        return v;
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(std::vector<int> new_order){
        Tensor result = makeStrideView(*this);
        result.inplace_permute(new_order);
        return result;
    }

    template <typename T>
    void Tensor<T>::inplace_permute(std::vector<int> new_order){
        check_inplace();
        if (new_order.size() != this->tensor_shape.shape.size()){
            throw std::invalid_argument("Permute order must have same length as tensor rank");
        }
        std::vector<bool> seen(this->tensor_shape.shape.size(), false);
        for (int i = 0; i < (int)new_order.size(); i++){
            if (new_order[i] < 0 || new_order[i] >= (int)this->tensor_shape.shape.size()){
                throw std::invalid_argument("Invalid permute order");
            }
            if (seen[new_order[i]]){
                throw std::invalid_argument("Permute order must be a permutation (no duplicates)");
            }
            seen[new_order[i]] = true;
        }

        std::vector<int> new_shape(this->tensor_shape.shape.size());
        std::vector<int> new_strides(this->tensor_shape.shape.size());
        std::vector<int> new_strides_col_major(this->tensor_shape.shape.size());
        for (int i = 0; i < (int)new_order.size(); i++){
            new_shape[i]             = this->tensor_shape.shape[new_order[i]];
            new_strides[i]           = this->tensor_shape.strides[new_order[i]];
            new_strides_col_major[i] = this->tensor_shape.strides_col_major[new_order[i]];
        }
        this->tensor_shape.shape             = std::move(new_shape);
        this->tensor_shape.strides           = std::move(new_strides);
        this->tensor_shape.strides_col_major = std::move(new_strides_col_major);
    }

    template <typename T>
    void Tensor<T>::reshape(std::vector<int> new_shape){
        check_inplace();
        // check multipliability
        if (utils::isSizeValid(new_shape, this->sizeOfTensor()) == false){
            throw std::invalid_argument("Invalid shape - needs to be multipliable to original shape");
        }
        this->tensor_shape.shape = new_shape;
        computeStrides();
    }

    template <typename T>
    void Tensor<T>::unsqueeze(int dim){
        check_inplace();
        std::vector<int> temp = this->tensor_shape.shape;
        temp.insert(temp.begin() + dim, 1);
        this->tensor_shape.shape = temp;
        computeStrides();
    }

    template <typename T>
    void Tensor<T>::squeeze(int dim){
        check_inplace();
        std::vector<int> temp = this->tensor_shape.shape;
        //debugging
        // std::cout << "INSIDE: " << std::endl;
        // for (size_t s = 0; s <temp.size(); s++){
        //     std::cout << temp[s] <<std::endl;
        // }

        if (temp.size() == 1){
            throw std::invalid_argument("Cannot squeeze dimension with size 1");
        }else{
            int tempDim = temp[dim];
            if (dim == 0){
                temp[dim+1]*=tempDim;
            }else{
                temp[dim-1]*=tempDim;
            }
            temp.erase(temp.begin() + dim);
            this->tensor_shape.shape = temp;
            computeStrides();
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::contiguous(const Tensor& t, const Device& device){
        if (t.is_contiguous()) return t; // already row-major + offset 0, nothing to do

        // result is a fresh row-major tensor with the same shape — its storage will be written
        // in linear order (data[0], data[1], ...) by the loop below
        Tensor result(t.tensor_shape.shape, t.device);
        const size_t n   = t.sizeOfTensor();
        const int    nd  = (int)t.tensor_shape.shape.size();

        if (t.device.type == DeviceType::CUDA) {
            // device kernel needs shape/strides on the device; copy them into size_t buffers and hand off
            std::vector<size_t> h_shape(nd);
            std::vector<size_t> h_strides(nd);
            for (int d = 0; d < nd; ++d) {
                h_shape[d]   = (size_t)t.tensor_shape.shape[d];
                h_strides[d] = (size_t)t.tensor_shape.strides[d];
            }
            cuda::utils::launch_contiguous_gather<cuda_type_trait_t<T>>(
                cuda_ptr(t.const_data()), cuda_ptr(result.mutable_data()), t.tensor_shape.data_offset,
                h_shape.data(), h_strides.data(), (size_t)nd, n);
        } else {
            // i walks the DESTINATION row-major (so we write result.data[i] in the natural flat order)
            for (size_t i = 0; i < n; ++i) {
                size_t tmp = i;               // running quotient — peels one axis at a time
                size_t src = t.tensor_shape.data_offset;   // running flat index into source storage; non-zero when t is a slice view
                // innermost dim varies fastest in row-major, so we modulo it out first and work outward
                for (int d = nd - 1; d >= 0; --d) {
                    size_t coord = tmp % (size_t)t.tensor_shape.shape[d];   // coord along axis d for this destination index
                    tmp /= (size_t)t.tensor_shape.shape[d];                  // strip that axis from tmp so the next iteration sees the outer dims
                    src += coord * (size_t)t.tensor_shape.strides[d];        // step in source storage by source's own stride along axis d
                                                                // -> for transpose this is the swapped dim's stride, for broadcast it's 0, for slice it's the parent's stride
                }
                result.at(i) = t.at(src); // dst is dense, so flat index == i; src lands wherever the source's strides put us
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::contiguous(const Device& device){
        if (is_contiguous()) return *this;
        return Tensor::contiguous(*this, device);
    }

    template <typename T>
    Tensor<T> Tensor<T>::slice(const Tensor& t, std::string parse){
        // We will have to compute a new shape and a new stride as well
        // [start0:end0, start1:end1, ...]

        // use python-like slice syntax
        // parse is a string that represents the slice operation
        // e.g. "0:10, 5:15" means slice the first 10 elements of the first dimension and the next 10 elements of the second dimension
        // parse the slice operation
        std::vector<std::string> sliceOps;
        boost::split(sliceOps, parse, boost::is_any_of(","));
        // check if the number of slice operations matches the number of dimensions
        if (sliceOps.size() != t.getShape().size()) {
            throw std::invalid_argument("Number of slice operations does not match number of dimensions.");
        }

        // parse the slice operations into start and end indices for each dimension
        size_t dim = 0;
        std::vector<utils::Slice> sliceIndices;
        for (const auto& op : sliceOps) {
            std::string cleaned = op;
            std::erase(cleaned, ' ');
            size_t dim_size = t.getShape()[dim];

            // check if the slice operation is valid
            if (cleaned ==":") {
                sliceIndices.emplace_back(0, dim_size, 1, utils::SliceMode::FULL);
            }else if (cleaned.find(':') == std::string::npos) {
                int idx = std::stoi(cleaned);
                sliceIndices.emplace_back(idx, idx + 1, 1, utils::SliceMode::RANGE);
            } else {
                std::vector<std::string> indices;
                boost::split(indices, cleaned, boost::is_any_of(":"));
                if (indices.size() > 3) throw std::invalid_argument("Invalid slice: " + cleaned);
                int start = indices[0].empty() ? 0       : std::stoi(indices[0]);
                int end   = indices[1].empty() ? dim_size : std::stoi(indices[1]);
                int step  = 1;
                if (indices.size() > 2 && !indices[2].empty()) step = std::stoi(indices[2]);
                if (start < 0 || start >= dim_size || end < start || end > dim_size || step <= 0)
                    throw std::invalid_argument("Invalid slice: " + cleaned);
                sliceIndices.emplace_back(start, end, step, utils::SliceMode::RANGE);
            }
            ++dim;

        }

        auto sliced_result = utils::computing_slice_parameters(t.getShape(), t.getStrides(), sliceIndices);

        Tensor slicedView = Tensor::makeSliceView(t, sliced_result);

        return slicedView;
    }

    template <typename T>
    Tensor<T> Tensor<T>::concat(std::initializer_list<Tensor> tensors, int dim){
        if (tensors.size() <= 1) {
            throw std::invalid_argument("More than one tensor is required for concatenation.");
        }
        std::vector<int> concatShapePrev = tensors.begin()[0].getShape();
        int concatShapeSize = concatShapePrev.size();
        int concatDim = concatShapePrev[dim];
        Device device = tensors.begin()[0].device;


        concatShapePrev[dim]= 0 ; // for comparison
        for (size_t i = 1; i < tensors.size(); ++i) {
            std::vector<int> copiedShape = tensors.begin()[i].getShape();
            copiedShape[dim] = 0;
            if (copiedShape != concatShapePrev){
                throw std::invalid_argument("Tensors must have the same shape for concatenation except along the concatenation dimension");
            }
            if (copiedShape.size() != concatShapeSize) {
                throw std::invalid_argument("Tensors must have the same shape for concatenation");
            }

            if (tensors.begin()[i].device != device) {
                throw std::invalid_argument("Tensors must have the same device for concatenation");
            }
            concatDim += tensors.begin()[i].getShape()[dim];
        }

        std::vector<int> temp = tensors.begin()[0].getShape();
        temp[dim] = concatDim;
        Tensor result(temp, tensors.begin()[0].device);


        // outer_dim, dim, inner_dim is what we have
        // copy the dim*inner_dim elements for each outer_dim (like the chunks)
        ll outerDim = 1;
        for (int i = 0; i < dim; ++i) {
            outerDim *= concatShapePrev[i];
        }

        ll innerDim = 1;
        for (int i = dim + 1; i < concatShapeSize; i++) {
            innerDim *= concatShapePrev[i];
        }

        if (device.type == DeviceType::CUDA) {
            const size_t n = tensors.size();

            std::vector<const T*> h_data(n);
            std::vector<int*> h_shape_ptrs(n); // each entry is a device pointer

            for (size_t i = 0; i < n; ++i) {
                const Tensor& t = tensors.begin()[i];
                h_data[i] = t.const_data(); // get the raw data pointer

                int* d_shape = nullptr;
                size_t shape_size = t.getShape().size();
                const size_t bytes = shape_size * sizeof(int);
                // allocate device memory for the shape and copy from host
                CUDA_CHECK(cudaMalloc(&d_shape, bytes));
                CUDA_CHECK(cudaMemcpy(d_shape, t.getShape().data(), bytes, cudaMemcpyHostToDevice));
                h_shape_ptrs[i] = d_shape;
            }

            const T** d_allInputs = nullptr;
            CUDA_CHECK(cudaMalloc(&d_allInputs, n * sizeof(T*)));
            CUDA_CHECK(cudaMemcpy(d_allInputs, h_data.data(),
                                  n * sizeof(T*), cudaMemcpyHostToDevice));

            // allocate the copied shapes pointers in a list of pointers on the device
            int** d_shapes = nullptr;
            CUDA_CHECK(cudaMalloc(&d_shapes, n * sizeof(int*)));
            CUDA_CHECK(cudaMemcpy(d_shapes, h_shape_ptrs.data(),
                                  n * sizeof(int*), cudaMemcpyHostToDevice));

            cuda::launch_concat_kernel<cuda_type_trait_t<T>>(
                reinterpret_cast<const cuda_type_trait_t<T>**>(d_allInputs), d_shapes, n, cuda_ptr(result.mutable_data()),
                outerDim, innerDim, dim, concatDim);

            for (int* d_shape : h_shape_ptrs) CUDA_CHECK(cudaFree(d_shape));
            CUDA_CHECK(cudaFree(d_allInputs));
            CUDA_CHECK(cudaFree(d_shapes));
        } else {
            for (size_t o = 0; o < outerDim; ++o) {
                int offset = 0; // offset will be used to move the pointer to execute the copies
                for (size_t i = 0; i < tensors.size(); ++i) {
                    int src_cat_dim = tensors.begin()[i].getShape()[dim];
                    int copy_size = src_cat_dim * innerDim;
                    // we copy the data for each tensor into the result tensor
                    // outer*concatDim * innerDim moves the pointer to the correct position in the result tensor
                    // offset * innerDim is the offset that will be copied from the source tensor using (o*src_cat_dim*innerDim)
                    std::memcpy(result.mutable_data()  + o*concatDim *innerDim + offset*innerDim, tensors.begin()[i].const_data() + o*src_cat_dim*innerDim, copy_size * sizeof(T));
                    offset += src_cat_dim;
                }
            }
        }

        // if (device.type == DeviceType::CUDA) {
        //     result.to_(device);
        // }

        return result;
    }

    template <typename T>
    void Tensor<T>::stack(std::initializer_list<Tensor> tensors){
        if (tensors.size() == 0) {
            throw std::invalid_argument("At least one tensor is required for stacking.");
        }
        // need to check that all tensors have the same shape - for stacking
        for (const Tensor& tensor : tensors){
            if (tensor.tensor_shape.shape != this->tensor_shape.shape){
                throw std::invalid_argument("Tensors must have the same shape for stacking");
            }
        }

        size_t stride = this->sizeOfTensor(); // captured before shape changes
        size_t total = (tensors.size()+1) * stride;
        auto new_storage = std::make_shared<Storage>(total * sizeof(T), this->device);

        this->tensor_shape.shape.insert(this->tensor_shape.shape.begin(), tensors.size()+1);
        computeStrides();

        // reads the old storage before storage is reassigned
        std::copy(this->mutable_data(), this->mutable_data() + stride, new_storage->data<T>());
        size_t start = stride;
        for (const Tensor& tensor : tensors){
            const T * tempData = tensor.const_data();

            std::copy(tempData, tempData + stride, new_storage->data<T>() + start); // copy the data
            start += stride;
        }

        this->storage = new_storage;
    }

}

template class bearml::Tensor<float>;
template class bearml::Tensor<double>;
template class bearml::Tensor<int>;
#if defined(__STDCPP_BFLOAT16_T__)
template class bearml::Tensor<std::bfloat16_t>;
#endif
