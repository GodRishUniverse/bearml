// Need to go over -> https://google.github.io/googletest/primer.html

#include "gtest/gtest.h"
#include "simplenet.h"

using namespace simplenet; // our namespace for simplenet types
using NodeT = std::shared_ptr<Node<TensorD>>; // aliases defined for ease of reading
using NodeD = std::shared_ptr<Node<double>>; // aliases defined for ease of reading

// Tensor Ops Tests

TEST(TensorTest, ConstructorZeroInitializes) {
    TensorD t({2, 3});
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            EXPECT_DOUBLE_EQ(t.get({r, c}), 0.0);
        }
    }
}

TEST(TensorTest, ShapeAndSize) {
    TensorD t({3, 4, 5});
    EXPECT_EQ(t.getShape(), (std::vector<int>{3, 4, 5}));
    EXPECT_EQ(t.sizeOfTensor(), 60u);
}

TEST(TensorTest, SetAndGet) {
    TensorD t({2, 2});
    t.set(3.14, {0, 1});
    EXPECT_DOUBLE_EQ(t.get({0, 1}), 3.14);
    EXPECT_DOUBLE_EQ(t.get({0, 0}), 0.0);
}

TEST(TensorTest, Fill) {
    TensorD t({2, 3});
    t.fill(7.0);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            EXPECT_DOUBLE_EQ(t.get({r, c}), 7.0);
        }
    }
}

TEST(TensorTest, Linspace) {
    TensorD t({5});
    t.linspace(0.0, 10.0);
    EXPECT_NEAR(t.get({0}), 0.0, 1e-10);
    EXPECT_NEAR(t.get({1}), 2.0, 1e-10);
    EXPECT_NEAR(t.get({4}), 8.0, 1e-10);
}

TEST(TensorTest, AddSameShape) {
    TensorD a({2, 2});
    TensorD b({2, 2});
    a.fill(1.0);
    b.fill(2.0);
    TensorD c = a + b;
    for (int r = 0; r < 2; r++) {
        for (int col = 0; col < 2; col++) {
            EXPECT_DOUBLE_EQ(c.get({r, col}), 3.0);
        }
    }
}

TEST(TensorTest, SubtractSameShape) {
    TensorD a({3});
    TensorD b({3});
    a.fill(5.0);
    b.fill(2.0);
    TensorD c = a - b;
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(c.get({i}), 3.0);
    }
}

TEST(TensorTest, ScalarMultiply) {
    TensorD a({2, 2});
    a.fill(3.0);
    TensorD c = a * 2.0;
    TensorD d = 2.0 * a;
    for (int r = 0; r < 2; r++) {
        for (int col = 0; col < 2; col++) {
            EXPECT_DOUBLE_EQ(c.get({r, col}), 6.0);
            EXPECT_DOUBLE_EQ(d.get({r, col}), 6.0);
        }
    }
}

TEST(TensorTest, ScalarDivide) {
    TensorD a({3});
    a.fill(6.0);
    TensorD c = a / 2.0;
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(c.get({i}), 3.0);
    }
}

TEST(TensorTest, ScalarAdd) {
    TensorD a({2});
    a.fill(1.0);
    TensorD c = a + 5.0;
    TensorD d = 5.0 + a;
    EXPECT_DOUBLE_EQ(c.get({0}), 6.0);
    EXPECT_DOUBLE_EQ(d.get({0}), 6.0);
}

TEST(TensorTest, ScalarSubtract) {
    TensorD a({2});
    a.fill(10.0);
    TensorD c = a - 3.0;
    TensorD d = 3.0 - a;
    EXPECT_DOUBLE_EQ(c.get({0}), 7.0);
    EXPECT_DOUBLE_EQ(d.get({0}), -7.0);
}

TEST(TensorTest, MatMul2x2) {
    TensorD a({2, 2});
    TensorD b({2, 2});
    // a = [[1,2],[3,4]]
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1});
    a.set(3.0, {1, 0}); a.set(4.0, {1, 1});
    // b = identity
    b.set(1.0, {0, 0}); b.set(0.0, {0, 1});
    b.set(0.0, {1, 0}); b.set(1.0, {1, 1});

    TensorD c = a * b;
    EXPECT_DOUBLE_EQ(c.get({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(c.get({0, 1}), 2.0);
    EXPECT_DOUBLE_EQ(c.get({1, 0}), 3.0);
    EXPECT_DOUBLE_EQ(c.get({1, 1}), 4.0);
}

TEST(TensorTest, MatMulValues) {
    TensorD a({2, 3});
    TensorD b({3, 2});
    // a = [[1,2,3],[4,5,6]]
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    // b = [[7,8],[9,10],[11,12]]
    b.set(7.0, {0, 0}); b.set(8.0, {0, 1});
    b.set(9.0, {1, 0}); b.set(10.0, {1, 1});
    b.set(11.0, {2, 0}); b.set(12.0, {2, 1});

    TensorD c = a * b; // [[58,64],[139,154]]
    EXPECT_DOUBLE_EQ(c.get({0, 0}), 58.0);
    EXPECT_DOUBLE_EQ(c.get({0, 1}), 64.0);
    EXPECT_DOUBLE_EQ(c.get({1, 0}), 139.0);
    EXPECT_DOUBLE_EQ(c.get({1, 1}), 154.0);
}

TEST(TensorTest, DotProduct) {
    TensorD a({3});
    TensorD b({3});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2});
    b.set(4.0, {0}); b.set(5.0, {1}); b.set(6.0, {2});
    TensorD c = a * b; // 1*4+2*5+3*6 = 32
    EXPECT_DOUBLE_EQ(c.get({0}), 32.0);
}

TEST(TensorTest, Transpose2D) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD t = a.transpose();
    EXPECT_EQ(t.getShape(), (std::vector<int>{3, 2}));
    EXPECT_DOUBLE_EQ(t.get({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(t.get({0, 1}), 4.0);
    EXPECT_DOUBLE_EQ(t.get({2, 0}), 3.0);
    EXPECT_DOUBLE_EQ(t.get({2, 1}), 6.0);
}

TEST(TensorTest, Reshape) {
    TensorD a({2, 3});
    a.linspace(1, 7);
    a.reshape({3, 2});
    EXPECT_EQ(a.getShape(), (std::vector<int>{3, 2}));
    EXPECT_NEAR(a.get({0, 0}), 1.0, 1e-10);
}

TEST(TensorTest, Unsqueeze) {
    TensorD a({3, 4});
    a.unsqueeze(0);
    EXPECT_EQ(a.getShape(), (std::vector<int>{1, 3, 4}));
}

TEST(TensorTest, Equality) {
    TensorD a({2, 2});
    TensorD b({2, 2});
    a.fill(1.0);
    b.fill(1.0);
    EXPECT_TRUE(a == b);
    b.set(2.0, {0, 0});
    EXPECT_TRUE(a != b);
}

TEST(TensorTest, Abs) {
    TensorD a({3});
    a.set(-3.0, {0}); a.set(2.0, {1}); a.set(-1.0, {2});
    TensorD b = TensorD::abs(a);
    EXPECT_DOUBLE_EQ(b.get({0}), 3.0);
    EXPECT_DOUBLE_EQ(b.get({1}), 2.0);
    EXPECT_DOUBLE_EQ(b.get({2}), 1.0);
}

TEST(TensorTest, Sqrt) {
    TensorD a({3});
    a.set(4.0, {0}); a.set(9.0, {1}); a.set(16.0, {2});
    TensorD b = TensorD::sqrt(a);
    EXPECT_DOUBLE_EQ(b.get({0}), 2.0);
    EXPECT_DOUBLE_EQ(b.get({1}), 3.0);
    EXPECT_DOUBLE_EQ(b.get({2}), 4.0);
}

TEST(TensorTest, Exp) {
    TensorD a({2});
    a.set(0.0, {0}); a.set(1.0, {1});
    TensorD b = TensorD::exp(a);
    EXPECT_NEAR(b.get({0}), 1.0, 1e-10);
    EXPECT_NEAR(b.get({1}), std::exp(1.0), 1e-10);
}

TEST(TensorTest, Log) {
    TensorD a({2});
    a.set(1.0, {0}); a.set(std::exp(2.0), {1});
    TensorD b = TensorD::log(a);
    EXPECT_NEAR(b.get({0}), 0.0, 1e-10);
    EXPECT_NEAR(b.get({1}), 2.0, 1e-10);
}

TEST(TensorTest, Mean) {
    TensorD a({4});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2}); a.set(4.0, {3});
    TensorD m = TensorD::mean(a);
    EXPECT_DOUBLE_EQ(m.get({0}), 2.5);
}

TEST(TensorTest, MaxElementWise) {
    TensorD a({3});
    TensorD b({3});
    a.set(1.0, {0}); a.set(5.0, {1}); a.set(3.0, {2});
    b.set(4.0, {0}); b.set(2.0, {1}); b.set(3.0, {2});
    TensorD c = TensorD::max(a, b);
    EXPECT_DOUBLE_EQ(c.get({0}), 4.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 5.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 3.0);
}

TEST(TensorTest, MinElementWise) {
    TensorD a({3});
    TensorD b({3});
    a.set(1.0, {0}); a.set(5.0, {1}); a.set(3.0, {2});
    b.set(4.0, {0}); b.set(2.0, {1}); b.set(3.0, {2});
    TensorD c = TensorD::min(a, b);
    EXPECT_DOUBLE_EQ(c.get({0}), 1.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 2.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 3.0);
}

TEST(TensorTest, Hadamard) {
    TensorD a({3});
    TensorD b({3});
    a.set(2.0, {0}); a.set(3.0, {1}); a.set(4.0, {2});
    b.set(5.0, {0}); b.set(6.0, {1}); b.set(7.0, {2});
    TensorD c = linear_algebra::hadamard(a, b);
    EXPECT_DOUBLE_EQ(c.get({0}), 10.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 18.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 28.0);
}

TEST(TensorTest, BroadcastAdd) {
    TensorD a({2, 3});
    TensorD b({1, 3});
    a.fill(1.0);
    b.set(10.0, {0, 0}); b.set(20.0, {0, 1}); b.set(30.0, {0, 2});
    TensorD c = a + b;
    EXPECT_EQ(c.getShape(), (std::vector<int>{2, 3}));
    EXPECT_DOUBLE_EQ(c.get({0, 0}), 11.0);
    EXPECT_DOUBLE_EQ(c.get({0, 1}), 21.0);
    EXPECT_DOUBLE_EQ(c.get({1, 2}), 31.0);
}

TEST(TensorTest, CopyConstructor) {
    TensorD a({2, 2});
    a.fill(5.0);
    TensorD b(a);
    b.set(99.0, {0, 0});
    EXPECT_DOUBLE_EQ(a.get({0, 0}), 5.0); // original unchanged
    EXPECT_DOUBLE_EQ(b.get({0, 0}), 99.0);
}

TEST(TensorTest, MoveConstructor) {
    TensorD a({2, 2});
    a.fill(3.0);
    TensorD b(std::move(a));
    EXPECT_DOUBLE_EQ(b.get({0, 0}), 3.0);
    EXPECT_EQ(b.getShape(), (std::vector<int>{2, 2}));
}

TEST(TensorTest, InvalidShapeThrows) {
    EXPECT_THROW(TensorD({0, 3}), std::invalid_argument);
    EXPECT_THROW(TensorD({-1, 3}), std::invalid_argument);
}

TEST(TensorTest, InvalidIndexThrows) {
    TensorD t({2, 2});
    EXPECT_THROW(t.get({0}), std::invalid_argument);     // wrong dims
    EXPECT_THROW(t.get({5, 0}), std::invalid_argument);  // out of range
}

TEST(TensorTest, ElementWiseDivide) {
    TensorD a({3});
    TensorD b({3});
    a.set(10.0, {0}); a.set(20.0, {1}); a.set(30.0, {2});
    b.set(2.0, {0}); b.set(5.0, {1}); b.set(10.0, {2});
    TensorD c = a / b;
    EXPECT_DOUBLE_EQ(c.get({0}), 5.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 4.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 3.0);
}

TEST(TensorTest, InPlaceAdd) {
    TensorD a({3});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2});
    TensorD b({3});
    b.fill(10.0);
    a += b;
    EXPECT_DOUBLE_EQ(a.get({0}), 11.0);
    EXPECT_DOUBLE_EQ(a.get({1}), 12.0);
    EXPECT_DOUBLE_EQ(a.get({2}), 13.0);
}

TEST(TensorTest, InPlaceSubtract) {
    TensorD a({2});
    a.fill(10.0);
    TensorD b({2});
    b.fill(3.0);
    a -= b;
    EXPECT_DOUBLE_EQ(a.get({0}), 7.0);
    EXPECT_DOUBLE_EQ(a.get({1}), 7.0);
}


// Autograd Double Tests

TEST(AutogradDoubleTest, AddBackward) {
    auto x = Node<double>::make_node(3.0);
    auto y = Node<double>::make_node(5.0);
    auto z = x + y; // z = x + y, dz/dx = 1, dz/dy = 1
    autogradient::backward(z);
    EXPECT_DOUBLE_EQ(x->grad, 1.0);
    EXPECT_DOUBLE_EQ(y->grad, 1.0);
}

TEST(AutogradDoubleTest, SubBackward) {
    auto x = Node<double>::make_node(3.0);
    auto y = Node<double>::make_node(5.0);
    auto z = x - y; // z = x - y, dz/dx = 1, dz/dy = -1
    autogradient::backward(z);
    EXPECT_DOUBLE_EQ(x->grad, 1.0);
    EXPECT_DOUBLE_EQ(y->grad, -1.0);
}

TEST(AutogradDoubleTest, MulBackward) {
    auto x = Node<double>::make_node(4.0);
    auto y = Node<double>::make_node(2.0);
    auto z = x * y; // z = x*y, dz/dx = y = 2, dz/dy = x = 4
    autogradient::backward(z);
    EXPECT_DOUBLE_EQ(x->grad, 2.0);
    EXPECT_DOUBLE_EQ(y->grad, 4.0);
}

TEST(AutogradDoubleTest, DivBackward) {
    auto x = Node<double>::make_node(6.0);
    auto z = x / 3.0; // z = x/3, dz/dx = 1/3
    autogradient::backward(z);
    EXPECT_NEAR(x->grad, 1.0 / 3.0, 1e-10);
}

TEST(AutogradDoubleTest, ExpBackward) {
    auto x = Node<double>::make_node(1.0);
    auto z = exp(x); // z = e^x, dz/dx = e^x = e
    autogradient::backward(z);
    EXPECT_NEAR(x->grad, std::exp(1.0), 1e-10);
}

TEST(AutogradDoubleTest, ChainRule) {
    auto x = Node<double>::make_node(4.0);
    auto y = Node<double>::make_node(2.0);
    auto z = x * y + x; // z = xy + x, dz/dx = y+1 = 3, dz/dy = x = 4
    autogradient::backward(z);
    EXPECT_DOUBLE_EQ(x->grad, 3.0);
    EXPECT_DOUBLE_EQ(y->grad, 4.0);
}

TEST(AutogradDoubleTest, MultipleOps) {
    auto x = Node<double>::make_node(2.0);
    auto y = Node<double>::make_node(3.0);
    auto a = x * y;     // a = 6
    auto b = a + x;     // b = 8, db/dx = y+1 = 4
    auto c = b * y;     // c = 24, dc/dx = (y+1)*y = 12, dc/dy = (xy+x) + x*y = 6+8 = ... let's just verify numerically
    autogradient::backward(c);
    // c = (x*y + x)*y = x*y^2 + x*y
    // dc/dx = y^2 + y = 9 + 3 = 12
    // dc/dy = 2xy + x = 12 + 2 = 14
    EXPECT_DOUBLE_EQ(x->grad, 12.0);
    EXPECT_DOUBLE_EQ(y->grad, 14.0);
}

// Autograd Tensor Tests

TEST(AutogradTensorTest, AddBackward) {
    TensorD ta({2, 2}); ta.fill(1.0);
    TensorD tb({2, 2}); tb.fill(2.0);
    auto a = Node<TensorD>::make_node(ta);
    auto b = Node<TensorD>::make_node(tb);
    auto c = a + b;
    autogradient::backward(c);
    // dc/da = 1, dc/db = 1
    TensorD ones({2, 2}); ones.fill(1.0);
    EXPECT_TRUE(a->grad == ones);
    EXPECT_TRUE(b->grad == ones);
}

TEST(AutogradTensorTest, SubBackward) {
    TensorD ta({2, 2}); ta.fill(3.0);
    TensorD tb({2, 2}); tb.fill(1.0);
    auto a = Node<TensorD>::make_node(ta);
    auto b = Node<TensorD>::make_node(tb);
    auto c = a - b;
    autogradient::backward(c);
    TensorD ones({2, 2}); ones.fill(1.0);
    TensorD neg_ones({2, 2}); neg_ones.fill(-1.0);
    EXPECT_TRUE(a->grad == ones);
    EXPECT_TRUE(b->grad == neg_ones);
}

TEST(AutogradTensorTest, MulBackward) {
    // matmul: C = A * B, dC/dA = grad * B^T, dC/dB = A^T * grad
    TensorD ta({2, 3}); ta.fill(1.0);
    TensorD tb({3, 2}); tb.fill(2.0);
    auto a = Node<TensorD>::make_node(ta);
    auto b = Node<TensorD>::make_node(tb);
    auto c = a * b; // matmul -> (2,2)
    autogradient::backward(c);
    // grad is ones(2,2)
    // dC/dA = ones(2,2) * B^T = ones(2,2) * 2*ones(2,3) = 2*ones * 2 cols... let's check shapes
    EXPECT_EQ(a->grad.getShape(), (std::vector<int>{2, 3}));
    EXPECT_EQ(b->grad.getShape(), (std::vector<int>{3, 2}));
}

TEST(AutogradTensorTest, HadamardBackward) {
    TensorD ta({3}); ta.set(2.0, {0}); ta.set(3.0, {1}); ta.set(4.0, {2});
    TensorD tb({3}); tb.set(5.0, {0}); tb.set(6.0, {1}); tb.set(7.0, {2});
    auto a = Node<TensorD>::make_node(ta);
    auto b = Node<TensorD>::make_node(tb);
    auto c = Node<TensorD>::hadamard(a, b);
    autogradient::backward(c);
    // d(a*b)/da = b, d(a*b)/db = a (element-wise, grad is ones)
    EXPECT_DOUBLE_EQ(a->grad.get({0}), 5.0);
    EXPECT_DOUBLE_EQ(a->grad.get({1}), 6.0);
    EXPECT_DOUBLE_EQ(a->grad.get({2}), 7.0);
    EXPECT_DOUBLE_EQ(b->grad.get({0}), 2.0);
    EXPECT_DOUBLE_EQ(b->grad.get({1}), 3.0);
    EXPECT_DOUBLE_EQ(b->grad.get({2}), 4.0);
}

TEST(AutogradTensorTest, AbsBackward) {
    TensorD ta({3}); ta.set(-2.0, {0}); ta.set(3.0, {1}); ta.set(-4.0, {2});
    auto a = Node<TensorD>::make_node(ta);
    auto c = abs(a);
    autogradient::backward(c);
    // d|x|/dx = sign(x)
    EXPECT_DOUBLE_EQ(a->grad.get({0}), -1.0);
    EXPECT_DOUBLE_EQ(a->grad.get({1}), 1.0);
    EXPECT_DOUBLE_EQ(a->grad.get({2}), -1.0);
}

TEST(AutogradTensorTest, MeanBackward) {
    TensorD ta({4}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(3.0, {2}); ta.set(4.0, {3});
    auto a = Node<TensorD>::make_node(ta);
    auto c = mean(a);
    autogradient::backward(c);
    // d(mean)/da_i = 1/n
    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(a->grad.get({i}), 0.25);
    }
}

TEST(AutogradTensorTest, TransposeBackward) {
    TensorD ta({2, 3}); ta.linspace(1.0, 7.0);
    auto a = Node<TensorD>::make_node(ta);
    auto c = transpose(a);
    EXPECT_EQ(c->val.getShape(), (std::vector<int>{3, 2}));
    autogradient::backward(c);
    // transpose backward is just transpose of grad
    EXPECT_EQ(a->grad.getShape(), (std::vector<int>{2, 3}));
}

TEST(AutogradTensorTest, ScalarNodeAdd) {
    TensorD ta({3}); ta.fill(2.0);
    auto a = Node<TensorD>::make_node(ta);
    auto c = a + 5.0;
    // c->val should be [7, 7, 7] with broadcasting
    autogradient::backward(c);
    // gradient of a should still flow
    EXPECT_TRUE(TensorD::has_nonzero_gradient(a->grad));
}

TEST(AutogradTensorTest, ScalarNodeSub) {
    TensorD ta({3}); ta.fill(10.0);
    auto a = Node<TensorD>::make_node(ta);
    auto c = 1.0 - a;
    autogradient::backward(c);
    // d(1-a)/da = -1
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(a->grad.get({i}), -1.0);
    }
}

TEST(AutogradTensorTest, ScalarNodeMul) {
    TensorD ta({3}); ta.fill(3.0);
    auto a = Node<TensorD>::make_node(ta);
    auto c = 2.0 * a;
    autogradient::backward(c);
    // d(2a)/da = 2
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(a->grad.get({i}), 2.0);
    }
}

TEST(AutogradTensorTest, DivBackward) {
    TensorD ta({3}); ta.fill(6.0);
    auto a = Node<TensorD>::make_node(ta);
    auto c = a / 3.0;
    autogradient::backward(c);
    // d(a/3)/da = 1/3
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(a->grad.get({i}), 1.0 / 3.0, 1e-10);
    }
}

TEST(AutogradTensorTest, LogBackward) {
    TensorD ta({3}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(4.0, {2});
    auto a = Node<TensorD>::make_node(ta);
    auto c = Node<TensorD>::log(a);
    autogradient::backward(c);
    // d(log(a))/da = 1/a
    EXPECT_NEAR(a->grad.get({0}), 1.0, 1e-10);
    EXPECT_NEAR(a->grad.get({1}), 0.5, 1e-10);
    EXPECT_NEAR(a->grad.get({2}), 0.25, 1e-10);
}

TEST(AutogradTensorTest, MaxBackward) {
    TensorD ta({3}); ta.set(1.0, {0}); ta.set(5.0, {1}); ta.set(3.0, {2});
    TensorD tb({3}); tb.set(4.0, {0}); tb.set(2.0, {1}); tb.set(3.0, {2});
    auto a = Node<TensorD>::make_node(ta);
    auto b = Node<TensorD>::make_node(tb);
    auto c = max(a, b);
    // max([1,5,3], [4,2,3]) = [4,5,3]
    EXPECT_DOUBLE_EQ(c->val.get({0}), 4.0);
    EXPECT_DOUBLE_EQ(c->val.get({1}), 5.0);
    EXPECT_DOUBLE_EQ(c->val.get({2}), 3.0);
    autogradient::backward(c);
    // grad goes to whichever was larger; ties go to both
    EXPECT_DOUBLE_EQ(a->grad.get({0}), 0.0); // b was larger
    EXPECT_DOUBLE_EQ(a->grad.get({1}), 1.0); // a was larger
    EXPECT_DOUBLE_EQ(a->grad.get({2}), 1.0); // tie
    EXPECT_DOUBLE_EQ(b->grad.get({0}), 1.0); // b was larger
    EXPECT_DOUBLE_EQ(b->grad.get({1}), 0.0); // a was larger
    EXPECT_DOUBLE_EQ(b->grad.get({2}), 1.0); // tie
}

TEST(AutogradTensorTest, ConstantNode) {
    TensorD ta({2}); ta.fill(3.0);
    auto a = Node<TensorD>::constant(ta);
    EXPECT_DOUBLE_EQ(a->val.get({0}), 3.0);
    EXPECT_DOUBLE_EQ(a->val.get({1}), 3.0);
    // constant nodes have no backward
    EXPECT_FALSE(a->backward_fn);
}


// Loss Function Tests

TEST(LossTest, L1LossZero) {
    TensorD ta({3}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(3.0, {2});
    auto actual = Node<TensorD>::make_node(ta);
    auto pred = Node<TensorD>::make_node(ta); // same values
    auto loss = neural_network::loss_functions::l1_loss(actual, pred);
    EXPECT_NEAR(loss->val.get({0}), 0.0, 1e-10);
}

TEST(LossTest, L1LossValue) {
    TensorD ta({4}); ta.fill(0.0);
    TensorD tp({4}); tp.set(1.0, {0}); tp.set(-1.0, {1}); tp.set(2.0, {2}); tp.set(-2.0, {3});
    auto actual = Node<TensorD>::make_node(ta);
    auto pred = Node<TensorD>::make_node(tp);
    auto loss = neural_network::loss_functions::l1_loss(actual, pred);
    // |1| + |-1| + |2| + |-2| = 6, mean = 1.5
    EXPECT_NEAR(loss->val.get({0}), 1.5, 1e-10);
}

TEST(LossTest, L2LossZero) {
    TensorD ta({3}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(3.0, {2});
    auto actual = Node<TensorD>::make_node(ta);
    auto pred = Node<TensorD>::make_node(ta);
    auto loss = neural_network::loss_functions::l2_loss(actual, pred);
    EXPECT_NEAR(loss->val.get({0}), 0.0, 1e-10);
}

TEST(LossTest, L2LossValue) {
    TensorD ta({3}); ta.fill(0.0);
    TensorD tp({3}); tp.set(1.0, {0}); tp.set(2.0, {1}); tp.set(3.0, {2});
    auto actual = Node<TensorD>::make_node(ta);
    auto pred = Node<TensorD>::make_node(tp);
    auto loss = neural_network::loss_functions::l2_loss(actual, pred);
    // (1+4+9)/3 = 14/3
    EXPECT_NEAR(loss->val.get({0}), 14.0 / 3.0, 1e-10);
}

TEST(LossTest, ShapeMismatchThrows) {
    TensorD ta({2}); ta.fill(1.0);
    TensorD tp({3}); tp.fill(1.0);
    auto a = Node<TensorD>::make_node(ta);
    auto p = Node<TensorD>::make_node(tp);
    EXPECT_THROW(neural_network::loss_functions::l1_loss(a, p), std::runtime_error);
    EXPECT_THROW(neural_network::loss_functions::l2_loss(a, p), std::runtime_error);
}


// Linear Layer Tests

TEST(LinearTest, OutputShape) {
    neural_network::Linear layer(5, 10);
    TensorD input({1, 5});
    input.fill(1.0);
    auto x = Node<TensorD>::make_node(input);
    auto out = layer(x);
    EXPECT_EQ(out->val.getShape(), (std::vector<int>{1, 10}));
}

TEST(LinearTest, Parameters) {
    neural_network::Linear layer(3, 4);
    auto params = layer.parameters();
    EXPECT_EQ(params.size(), 2u); // weight + bias
    EXPECT_EQ(params[0]->val.getShape(), (std::vector<int>{3, 4})); // weight
    EXPECT_EQ(params[1]->val.getShape(), (std::vector<int>{4})); // bias
}

TEST(LinearTest, BackwardProducesGradients) {
    neural_network::Linear layer(3, 2);
    TensorD input({1, 3});
    input.fill(1.0);
    auto x = Node<TensorD>::make_node(input);
    auto out = layer(x);
    autogradient::backward(out);
    auto params = layer.parameters();
    EXPECT_TRUE(TensorD::has_nonzero_gradient(params[0]->grad)); // weight has grad
    EXPECT_TRUE(TensorD::has_nonzero_gradient(params[1]->grad)); // bias has grad
}


// ReLU Tests

TEST(ReLUTest, ForwardPositive) {
    neural_network::ReLU relu;
    TensorD input({4});
    input.set(-2.0, {0}); input.set(0.0, {1}); input.set(3.0, {2}); input.set(-1.0, {3});
    auto x = Node<TensorD>::make_node(input);
    auto out = relu(x);
    EXPECT_DOUBLE_EQ(out->val.get({0}), 0.0);
    EXPECT_DOUBLE_EQ(out->val.get({1}), 0.0);
    EXPECT_DOUBLE_EQ(out->val.get({2}), 3.0);
    EXPECT_DOUBLE_EQ(out->val.get({3}), 0.0);
}


// SGD Optimizer Tests

TEST(SGDTest, StepUpdatesParams) {
    TensorD tw({2}); tw.fill(5.0);
    auto w = Node<TensorD>::make_node(tw);
    w->grad = TensorD({2}); w->grad.fill(1.0); // manually set grad

    neural_network::optimizers::SGD sgd({w}, 0.1);
    sgd.step();
    // w = w - lr * grad = 5 - 0.1*1 = 4.9
    EXPECT_NEAR(w->val.get({0}), 4.9, 1e-10);
    EXPECT_NEAR(w->val.get({1}), 4.9, 1e-10);
}

TEST(SGDTest, ZeroGrad) {
    TensorD tw({3}); tw.fill(1.0);
    auto w = Node<TensorD>::make_node(tw);
    w->grad = TensorD({3}); w->grad.fill(5.0);

    neural_network::optimizers::SGD sgd({w}, 0.01);
    sgd.zero_grad();
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(w->grad.get({i}), 0.0);
    }
}


// Reduction Tests

// --- SUM ---

TEST(ReductionTest, Sum1D) {
    TensorD a({5});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2}); a.set(4.0, {3}); a.set(5.0, {4});
    TensorD r = a.accumulate(0, reductions::ReductionOps::SUM);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 15.0);
}

TEST(ReductionTest, Sum2D_Dim0) {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::SUM, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 5.0);  // 1+4
    EXPECT_DOUBLE_EQ(r.get({1}), 7.0);  // 2+5
    EXPECT_DOUBLE_EQ(r.get({2}), 9.0);  // 3+6
}

TEST(ReductionTest, Sum2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::SUM, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 6.0);   // 1+2+3
    EXPECT_DOUBLE_EQ(r.get({1}), 15.0);  // 4+5+6
}

TEST(ReductionTest, Sum2D_Dim0_KeepDims) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::SUM, true);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1, 3}));
    EXPECT_DOUBLE_EQ(r.get({0, 0}), 5.0);
    EXPECT_DOUBLE_EQ(r.get({0, 1}), 7.0);
    EXPECT_DOUBLE_EQ(r.get({0, 2}), 9.0);
}

// --- MEAN ---

TEST(ReductionTest, Mean1D) {
    TensorD a({4});
    a.set(2.0, {0}); a.set(4.0, {1}); a.set(6.0, {2}); a.set(8.0, {3});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MEAN);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 5.0);
}

TEST(ReductionTest, Mean2D_Dim0) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MEAN, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 2.5);  // (1+4)/2
    EXPECT_DOUBLE_EQ(r.get({1}), 3.5);  // (2+5)/2
    EXPECT_DOUBLE_EQ(r.get({2}), 4.5);  // (3+6)/2
}

TEST(ReductionTest, Mean2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::MEAN, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 2.0);  // (1+2+3)/3
    EXPECT_DOUBLE_EQ(r.get({1}), 5.0);  // (4+5+6)/3
}

TEST(ReductionTest, Mean2D_Dim1_KeepDims) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::MEAN, true);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2, 1}));
    EXPECT_DOUBLE_EQ(r.get({0, 0}), 2.0);
    EXPECT_DOUBLE_EQ(r.get({1, 0}), 5.0);
}

// --- MAX ---

TEST(ReductionTest, Max1D) {
    TensorD a({5});
    a.set(3.0, {0}); a.set(1.0, {1}); a.set(7.0, {2}); a.set(2.0, {3}); a.set(5.0, {4});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MAX);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 7.0);
}

TEST(ReductionTest, Max2D_Dim0) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MAX, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 4.0);  // max(1,4)
    EXPECT_DOUBLE_EQ(r.get({1}), 5.0);  // max(5,2)
    EXPECT_DOUBLE_EQ(r.get({2}), 6.0);  // max(3,6)
}

TEST(ReductionTest, Max2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::MAX, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 5.0);  // max(1,5,3)
    EXPECT_DOUBLE_EQ(r.get({1}), 6.0);  // max(4,2,6)
}

TEST(ReductionTest, Max1D_Negative) {
    TensorD a({3});
    a.set(-5.0, {0}); a.set(-1.0, {1}); a.set(-3.0, {2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MAX);
    EXPECT_DOUBLE_EQ(r.get({0}), -1.0);
}

// --- MIN ---

TEST(ReductionTest, Min1D) {
    TensorD a({5});
    a.set(3.0, {0}); a.set(1.0, {1}); a.set(7.0, {2}); a.set(2.0, {3}); a.set(5.0, {4});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MIN);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 1.0);
}

TEST(ReductionTest, Min2D_Dim0) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MIN, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 1.0);  // min(1,4)
    EXPECT_DOUBLE_EQ(r.get({1}), 2.0);  // min(5,2)
    EXPECT_DOUBLE_EQ(r.get({2}), 3.0);  // min(3,6)
}

TEST(ReductionTest, Min2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::MIN, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 1.0);  // min(1,5,3)
    EXPECT_DOUBLE_EQ(r.get({1}), 2.0);  // min(4,2,6)
}

TEST(ReductionTest, Min1D_Negative) {
    TensorD a({3});
    a.set(-5.0, {0}); a.set(-1.0, {1}); a.set(-3.0, {2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::MIN);
    EXPECT_DOUBLE_EQ(r.get({0}), -5.0);
}

// --- PROD ---

TEST(ReductionTest, Prod1D) {
    TensorD a({4});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2}); a.set(4.0, {3});
    TensorD r = a.accumulate(0, reductions::ReductionOps::PROD);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 24.0);  // 1*2*3*4
}

TEST(ReductionTest, Prod2D_Dim0) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::PROD, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 4.0);   // 1*4
    EXPECT_DOUBLE_EQ(r.get({1}), 10.0);  // 2*5
    EXPECT_DOUBLE_EQ(r.get({2}), 18.0);  // 3*6
}

TEST(ReductionTest, Prod2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::PROD, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 6.0);    // 1*2*3
    EXPECT_DOUBLE_EQ(r.get({1}), 120.0);  // 4*5*6
}

TEST(ReductionTest, Prod1D_WithZero) {
    TensorD a({3});
    a.set(5.0, {0}); a.set(0.0, {1}); a.set(3.0, {2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::PROD);
    EXPECT_DOUBLE_EQ(r.get({0}), 0.0);
}

TEST(ReductionTest, Prod2D_Dim0_KeepDims) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::PROD, true);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1, 3}));
    EXPECT_DOUBLE_EQ(r.get({0, 0}), 4.0);
    EXPECT_DOUBLE_EQ(r.get({0, 1}), 10.0);
    EXPECT_DOUBLE_EQ(r.get({0, 2}), 18.0);
}

// --- ARG_MAX ---

TEST(ReductionTest, ArgMax1D) {
    TensorD a({5});
    a.set(3.0, {0}); a.set(1.0, {1}); a.set(7.0, {2}); a.set(2.0, {3}); a.set(5.0, {4});
    TensorD r = a.accumulate(0, reductions::ReductionOps::ARG_MAX);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 2.0);  // index of 7.0
}

TEST(ReductionTest, ArgMax2D_Dim0) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::ARG_MAX, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 1.0);  // row 1 has 4 > 1
    EXPECT_DOUBLE_EQ(r.get({1}), 0.0);  // row 0 has 5 > 2
    EXPECT_DOUBLE_EQ(r.get({2}), 1.0);  // row 1 has 6 > 3
}

TEST(ReductionTest, ArgMax2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::ARG_MAX, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 1.0);  // col 1 has max 5
    EXPECT_DOUBLE_EQ(r.get({1}), 2.0);  // col 2 has max 6
}

// --- ARG_MIN ---

TEST(ReductionTest, ArgMin1D) {
    TensorD a({5});
    a.set(3.0, {0}); a.set(1.0, {1}); a.set(7.0, {2}); a.set(2.0, {3}); a.set(5.0, {4});
    TensorD r = a.accumulate(0, reductions::ReductionOps::ARG_MIN);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(r.get({0}), 1.0);  // index of 1.0
}

TEST(ReductionTest, ArgMin2D_Dim0) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::ARG_MIN, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{3}));
    EXPECT_DOUBLE_EQ(r.get({0}), 0.0);  // row 0 has 1 < 4
    EXPECT_DOUBLE_EQ(r.get({1}), 1.0);  // row 1 has 2 < 5
    EXPECT_DOUBLE_EQ(r.get({2}), 0.0);  // row 0 has 3 < 6
}

TEST(ReductionTest, ArgMin2D_Dim1) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(5.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(2.0, {1, 1}); a.set(6.0, {1, 2});
    TensorD r = a.accumulate(1, reductions::ReductionOps::ARG_MIN, false);
    EXPECT_EQ(r.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(r.get({0}), 0.0);  // col 0 has min 1
    EXPECT_DOUBLE_EQ(r.get({1}), 1.0);  // col 1 has min 2
}

// --- Edge cases ---

TEST(ReductionTest, SingleElementTensor) {
    TensorD a({1});
    a.set(42.0, {0});
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::SUM).get({0}), 42.0);
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::MEAN).get({0}), 42.0);
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::MAX).get({0}), 42.0);
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::MIN).get({0}), 42.0);
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::PROD).get({0}), 42.0);
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::ARG_MAX).get({0}), 0.0);
    EXPECT_DOUBLE_EQ(a.accumulate(0, reductions::ReductionOps::ARG_MIN).get({0}), 0.0);
}

TEST(ReductionTest, InvalidDimThrows) {
    TensorD a({2, 3});
    EXPECT_THROW(a.accumulate(2, reductions::ReductionOps::SUM), std::invalid_argument);
    EXPECT_THROW(a.accumulate(-1, reductions::ReductionOps::SUM), std::invalid_argument);
}

TEST(ReductionTest, Reduce3D_SumDim0) {
    // shape [2, 2, 3]
    TensorD a({2, 2, 3});
    // layer 0: [[1,2,3],[4,5,6]]
    a.set(1.0, {0, 0, 0}); a.set(2.0, {0, 0, 1}); a.set(3.0, {0, 0, 2});
    a.set(4.0, {0, 1, 0}); a.set(5.0, {0, 1, 1}); a.set(6.0, {0, 1, 2});
    // layer 1: [[7,8,9],[10,11,12]]
    a.set(7.0, {1, 0, 0}); a.set(8.0, {1, 0, 1}); a.set(9.0, {1, 0, 2});
    a.set(10.0, {1, 1, 0}); a.set(11.0, {1, 1, 1}); a.set(12.0, {1, 1, 2});
    TensorD r = a.accumulate(0, reductions::ReductionOps::SUM, false);
    // result shape [2, 3]: [[1+7, 2+8, 3+9], [4+10, 5+11, 6+12]]
    EXPECT_EQ(r.getShape(), (std::vector<int>{2, 3}));
    EXPECT_DOUBLE_EQ(r.get({0, 0}), 8.0);
    EXPECT_DOUBLE_EQ(r.get({0, 1}), 10.0);
    EXPECT_DOUBLE_EQ(r.get({0, 2}), 12.0);
    EXPECT_DOUBLE_EQ(r.get({1, 0}), 14.0);
    EXPECT_DOUBLE_EQ(r.get({1, 1}), 16.0);
    EXPECT_DOUBLE_EQ(r.get({1, 2}), 18.0);
}

TEST(ReductionTest, ReduceViaLinalgReduce) {
    TensorD a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    // reduce to shape {1, 1} (full reduction via SUM)
    std::vector<int> targetShape = {1, 1};
    TensorD r = linear_algebra::reduce(a, targetShape, reductions::ReductionOps::SUM);
    EXPECT_EQ(r.getShape(), (std::vector<int>{1, 1}));
    EXPECT_DOUBLE_EQ(r.get({0, 0}), 21.0);  // 1+2+3+4+5+6
}


// main to run all the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
