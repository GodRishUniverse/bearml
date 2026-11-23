// acceptance tests for the lazy IR - they fail with "not implemented" until the stubs are filled in

#include "gtest/gtest.h"
#include "bearml.h"

using namespace bearml;

// IRBuilder Tests

TEST(IRBuilder, ParameterCarriesShapeDtypeDevice) {
    IRBuilder b;
    auto p = b.parameter(Tensorf({4, 3}));
    EXPECT_EQ(p->node_type, IRNodeType::PARAMETER);
    EXPECT_EQ(p->shape, (std::vector<int>{4, 3}));
    EXPECT_EQ(p->dtype, DType::F32);
    EXPECT_TRUE(p->device.is_cpu());
    EXPECT_TRUE(p->inputs.empty());
    EXPECT_NE(p->leaf_value, nullptr);
}

TEST(IRBuilder, ParametersAreNeverMerged) {
    IRBuilder b;
    Tensorf t({2, 2});
    auto p1 = b.parameter(t);
    auto p2 = b.parameter(t);
    EXPECT_NE(p1, p2);
    EXPECT_NE(p1->id, p2->id);
}

TEST(IRBuilder, CSEReturnsSameNode) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({2, 2}));
    auto m1 = b.op(OP_Code::OP_MUL_SCALAR, {x}, OpAttributes{.constant = 2.0});
    auto m2 = b.op(OP_Code::OP_MUL_SCALAR, {x}, OpAttributes{.constant = 2.0});
    EXPECT_EQ(m1, m2);
    EXPECT_EQ(b.num_nodes(), 2u);
}

TEST(IRBuilder, CSEKeepsDifferentAttrsApart) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({2, 2}));
    auto m2 = b.op(OP_Code::OP_MUL_SCALAR, {x}, OpAttributes{.constant = 2.0});
    auto m3 = b.op(OP_Code::OP_MUL_SCALAR, {x}, OpAttributes{.constant = 3.0});
    EXPECT_NE(m2, m3);
}

// InferType Tests

TEST(InferType, BinaryBroadcasts) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({4, 1}));
    auto y = b.parameter(Tensorf({1, 8}));
    InferredType t = infer_type(OP_Code::OP_ADD, {x, y}, {});
    EXPECT_EQ(t.shape, (std::vector<int>{4, 8}));
    EXPECT_EQ(t.dtype, DType::F32);
}

TEST(InferType, BinaryShapeMismatchThrows) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({4, 3}));
    auto y = b.parameter(Tensorf({5, 3}));
    EXPECT_THROW(infer_type(OP_Code::OP_ADD, {x, y}, {}), std::invalid_argument);
}

TEST(InferType, DtypeMismatchThrows) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({2, 2}));
    auto y = b.parameter(TensorD({2, 2}));
    EXPECT_THROW(infer_type(OP_Code::OP_ADD, {x, y}, {}), std::invalid_argument);
}

TEST(InferType, TransposeSwapsLastTwoDims) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({2, 3, 5}));
    EXPECT_EQ(infer_type(OP_Code::OP_TRANSPOSE, {x}, {}).shape, (std::vector<int>{2, 5, 3}));
}

TEST(InferType, SumKeepsReducedDimAsOne) {
    IRBuilder b;
    auto x = b.parameter(Tensorf({4, 6}));
    EXPECT_EQ(infer_type(OP_Code::OP_SUM, {x}, OpAttributes{.dim = 1}).shape, (std::vector<int>{4, 1}));
}

// LazyTensor Tests

TEST(LazyTensor, OpsBuildGraphWithoutComputing) {
    auto a = LazyTensor::from(Tensorf({2, 3}));
    auto b = LazyTensor::from(Tensorf({2, 3}));
    auto c = a + b;
    EXPECT_EQ(c.ir()->op, OP_Code::OP_ADD);
    ASSERT_EQ(c.ir()->inputs.size(), 2u);
    EXPECT_EQ(c.ir()->inputs[0], a.ir());
    EXPECT_EQ(c.ir()->inputs[1], b.ir());
    EXPECT_EQ(c.ir()->leaf_value, nullptr);
    EXPECT_EQ(c.shape(), (std::vector<int>{2, 3}));
}

TEST(LazyTensor, ToStringShowsIR) {
    auto a = LazyTensor::from(Tensorf({2, 3}));
    auto c = LazyTensor::exp(a);
    std::string s = c.to_string();
    EXPECT_NE(s.find("OP_EXP"), std::string::npos);
    EXPECT_NE(s.find("%" + std::to_string(a.ir()->id)), std::string::npos);
}

// enable once an interpreter exists
TEST(LazyTensor, DISABLED_ToTensorMatchesEager) {
    TensorD x({2, 2});
    x.fill(1.5);
    TensorD eager = TensorD::exp(x) * 2.0;
    TensorD lazy = (LazyTensor::exp(LazyTensor::from(x)) * 2.0).to_tensor<double>();
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            EXPECT_DOUBLE_EQ(lazy.get({r, c}), eager.get({r, c}));
        }
    }
}
