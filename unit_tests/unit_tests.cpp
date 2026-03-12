// Need to go over -> https://google.github.io/googletest/primer.html

#include "gtest/gtest.h"
#include "simplenet.h"

using namespace simplenet; // our namespace for simplenet types
using NodeT = std::shared_ptr<Node<Tensor>>; // aliases defined for ease of reading
using NodeD = std::shared_ptr<Node<double>>; // aliases defined for ease of reading

// Tensor Ops Tests

TEST(TensorTest, ConstructorZeroInitializes) {
    Tensor t({2, 3});
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            EXPECT_DOUBLE_EQ(t.get({r, c}), 0.0);
        }
    }
}

TEST(TensorTest, ShapeAndSize) {
    Tensor t({3, 4, 5});
    EXPECT_EQ(t.getShape(), (std::vector<int>{3, 4, 5}));
    EXPECT_EQ(t.sizeOfTensor(), 60u);
}

TEST(TensorTest, SetAndGet) {
    Tensor t({2, 2});
    t.set(3.14, {0, 1});
    EXPECT_DOUBLE_EQ(t.get({0, 1}), 3.14);
    EXPECT_DOUBLE_EQ(t.get({0, 0}), 0.0);
}

TEST(TensorTest, Fill) {
    Tensor t({2, 3});
    t.fill(7.0);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            EXPECT_DOUBLE_EQ(t.get({r, c}), 7.0);
        }
    }
}

TEST(TensorTest, Linspace) {
    Tensor t({5});
    t.linspace(0.0, 10.0);
    EXPECT_NEAR(t.get({0}), 0.0, 1e-10);
    EXPECT_NEAR(t.get({1}), 2.0, 1e-10);
    EXPECT_NEAR(t.get({4}), 8.0, 1e-10);
}

TEST(TensorTest, AddSameShape) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(1.0);
    b.fill(2.0);
    Tensor c = a + b;
    for (int r = 0; r < 2; r++) {
        for (int col = 0; col < 2; col++) {
            EXPECT_DOUBLE_EQ(c.get({r, col}), 3.0);
        }
    }
}

TEST(TensorTest, SubtractSameShape) {
    Tensor a({3});
    Tensor b({3});
    a.fill(5.0);
    b.fill(2.0);
    Tensor c = a - b;
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(c.get({i}), 3.0);
    }
}

TEST(TensorTest, ScalarMultiply) {
    Tensor a({2, 2});
    a.fill(3.0);
    Tensor c = a * 2.0;
    Tensor d = 2.0 * a;
    for (int r = 0; r < 2; r++) {
        for (int col = 0; col < 2; col++) {
            EXPECT_DOUBLE_EQ(c.get({r, col}), 6.0);
            EXPECT_DOUBLE_EQ(d.get({r, col}), 6.0);
        }
    }
}

TEST(TensorTest, ScalarDivide) {
    Tensor a({3});
    a.fill(6.0);
    Tensor c = a / 2.0;
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(c.get({i}), 3.0);
    }
}

TEST(TensorTest, ScalarAdd) {
    Tensor a({2});
    a.fill(1.0);
    Tensor c = a + 5.0;
    Tensor d = 5.0 + a;
    EXPECT_DOUBLE_EQ(c.get({0}), 6.0);
    EXPECT_DOUBLE_EQ(d.get({0}), 6.0);
}

TEST(TensorTest, ScalarSubtract) {
    Tensor a({2});
    a.fill(10.0);
    Tensor c = a - 3.0;
    Tensor d = 3.0 - a;
    EXPECT_DOUBLE_EQ(c.get({0}), 7.0);
    EXPECT_DOUBLE_EQ(d.get({0}), -7.0);
}

TEST(TensorTest, MatMul2x2) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    // a = [[1,2],[3,4]]
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1});
    a.set(3.0, {1, 0}); a.set(4.0, {1, 1});
    // b = identity
    b.set(1.0, {0, 0}); b.set(0.0, {0, 1});
    b.set(0.0, {1, 0}); b.set(1.0, {1, 1});

    Tensor c = a * b;
    EXPECT_DOUBLE_EQ(c.get({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(c.get({0, 1}), 2.0);
    EXPECT_DOUBLE_EQ(c.get({1, 0}), 3.0);
    EXPECT_DOUBLE_EQ(c.get({1, 1}), 4.0);
}

TEST(TensorTest, MatMulValues) {
    Tensor a({2, 3});
    Tensor b({3, 2});
    // a = [[1,2,3],[4,5,6]]
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    // b = [[7,8],[9,10],[11,12]]
    b.set(7.0, {0, 0}); b.set(8.0, {0, 1});
    b.set(9.0, {1, 0}); b.set(10.0, {1, 1});
    b.set(11.0, {2, 0}); b.set(12.0, {2, 1});

    Tensor c = a * b; // [[58,64],[139,154]]
    EXPECT_DOUBLE_EQ(c.get({0, 0}), 58.0);
    EXPECT_DOUBLE_EQ(c.get({0, 1}), 64.0);
    EXPECT_DOUBLE_EQ(c.get({1, 0}), 139.0);
    EXPECT_DOUBLE_EQ(c.get({1, 1}), 154.0);
}

TEST(TensorTest, DotProduct) {
    Tensor a({3});
    Tensor b({3});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2});
    b.set(4.0, {0}); b.set(5.0, {1}); b.set(6.0, {2});
    Tensor c = a * b; // 1*4+2*5+3*6 = 32
    EXPECT_DOUBLE_EQ(c.get({0}), 32.0);
}

TEST(TensorTest, Transpose2D) {
    Tensor a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    Tensor t = a.transpose();
    EXPECT_EQ(t.getShape(), (std::vector<int>{3, 2}));
    EXPECT_DOUBLE_EQ(t.get({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(t.get({0, 1}), 4.0);
    EXPECT_DOUBLE_EQ(t.get({2, 0}), 3.0);
    EXPECT_DOUBLE_EQ(t.get({2, 1}), 6.0);
}

TEST(TensorTest, Reshape) {
    Tensor a({2, 3});
    a.linspace(1, 7);
    a.reshape({3, 2});
    EXPECT_EQ(a.getShape(), (std::vector<int>{3, 2}));
    EXPECT_NEAR(a.get({0, 0}), 1.0, 1e-10);
}

TEST(TensorTest, Unsqueeze) {
    Tensor a({3, 4});
    a.unsqueeze(0);
    EXPECT_EQ(a.getShape(), (std::vector<int>{1, 3, 4}));
}

TEST(TensorTest, Equality) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(1.0);
    b.fill(1.0);
    EXPECT_TRUE(a == b);
    b.set(2.0, {0, 0});
    EXPECT_TRUE(a != b);
}

TEST(TensorTest, Abs) {
    Tensor a({3});
    a.set(-3.0, {0}); a.set(2.0, {1}); a.set(-1.0, {2});
    Tensor b = Tensor::abs(a);
    EXPECT_DOUBLE_EQ(b.get({0}), 3.0);
    EXPECT_DOUBLE_EQ(b.get({1}), 2.0);
    EXPECT_DOUBLE_EQ(b.get({2}), 1.0);
}

TEST(TensorTest, Sqrt) {
    Tensor a({3});
    a.set(4.0, {0}); a.set(9.0, {1}); a.set(16.0, {2});
    Tensor b = Tensor::sqrt(a);
    EXPECT_DOUBLE_EQ(b.get({0}), 2.0);
    EXPECT_DOUBLE_EQ(b.get({1}), 3.0);
    EXPECT_DOUBLE_EQ(b.get({2}), 4.0);
}

TEST(TensorTest, Exp) {
    Tensor a({2});
    a.set(0.0, {0}); a.set(1.0, {1});
    Tensor b = Tensor::exp(a);
    EXPECT_NEAR(b.get({0}), 1.0, 1e-10);
    EXPECT_NEAR(b.get({1}), std::exp(1.0), 1e-10);
}

TEST(TensorTest, Log) {
    Tensor a({2});
    a.set(1.0, {0}); a.set(std::exp(2.0), {1});
    Tensor b = Tensor::log(a);
    EXPECT_NEAR(b.get({0}), 0.0, 1e-10);
    EXPECT_NEAR(b.get({1}), 2.0, 1e-10);
}

TEST(TensorTest, Mean) {
    Tensor a({4});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2}); a.set(4.0, {3});
    Tensor m = Tensor::mean(a);
    EXPECT_DOUBLE_EQ(m.get({0}), 2.5);
}

TEST(TensorTest, MaxElementWise) {
    Tensor a({3});
    Tensor b({3});
    a.set(1.0, {0}); a.set(5.0, {1}); a.set(3.0, {2});
    b.set(4.0, {0}); b.set(2.0, {1}); b.set(3.0, {2});
    Tensor c = Tensor::max(a, b);
    EXPECT_DOUBLE_EQ(c.get({0}), 4.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 5.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 3.0);
}

TEST(TensorTest, MinElementWise) {
    Tensor a({3});
    Tensor b({3});
    a.set(1.0, {0}); a.set(5.0, {1}); a.set(3.0, {2});
    b.set(4.0, {0}); b.set(2.0, {1}); b.set(3.0, {2});
    Tensor c = Tensor::min(a, b);
    EXPECT_DOUBLE_EQ(c.get({0}), 1.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 2.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 3.0);
}

TEST(TensorTest, Hadamard) {
    Tensor a({3});
    Tensor b({3});
    a.set(2.0, {0}); a.set(3.0, {1}); a.set(4.0, {2});
    b.set(5.0, {0}); b.set(6.0, {1}); b.set(7.0, {2});
    Tensor c = linear_algebra::hadamard(a, b);
    EXPECT_DOUBLE_EQ(c.get({0}), 10.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 18.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 28.0);
}

TEST(TensorTest, BroadcastAdd) {
    Tensor a({2, 3});
    Tensor b({1, 3});
    a.fill(1.0);
    b.set(10.0, {0, 0}); b.set(20.0, {0, 1}); b.set(30.0, {0, 2});
    Tensor c = a + b;
    EXPECT_EQ(c.getShape(), (std::vector<int>{2, 3}));
    EXPECT_DOUBLE_EQ(c.get({0, 0}), 11.0);
    EXPECT_DOUBLE_EQ(c.get({0, 1}), 21.0);
    EXPECT_DOUBLE_EQ(c.get({1, 2}), 31.0);
}

TEST(TensorTest, SumDim) {
    Tensor a({2, 3});
    a.set(1.0, {0, 0}); a.set(2.0, {0, 1}); a.set(3.0, {0, 2});
    a.set(4.0, {1, 0}); a.set(5.0, {1, 1}); a.set(6.0, {1, 2});
    // sum along dim 1: [6, 15]
    Tensor s = a.sum(1);
    EXPECT_EQ(s.getShape(), (std::vector<int>{2}));
    EXPECT_DOUBLE_EQ(s.get({0}), 6.0);
    EXPECT_DOUBLE_EQ(s.get({1}), 15.0);
}

TEST(TensorTest, CopyConstructor) {
    Tensor a({2, 2});
    a.fill(5.0);
    Tensor b(a);
    b.set(99.0, {0, 0});
    EXPECT_DOUBLE_EQ(a.get({0, 0}), 5.0); // original unchanged
    EXPECT_DOUBLE_EQ(b.get({0, 0}), 99.0);
}

TEST(TensorTest, MoveConstructor) {
    Tensor a({2, 2});
    a.fill(3.0);
    Tensor b(std::move(a));
    EXPECT_DOUBLE_EQ(b.get({0, 0}), 3.0);
    EXPECT_EQ(b.getShape(), (std::vector<int>{2, 2}));
}

TEST(TensorTest, InvalidShapeThrows) {
    EXPECT_THROW(Tensor({0, 3}), std::invalid_argument);
    EXPECT_THROW(Tensor({-1, 3}), std::invalid_argument);
}

TEST(TensorTest, InvalidIndexThrows) {
    Tensor t({2, 2});
    EXPECT_THROW(t.get({0}), std::invalid_argument);     // wrong dims
    EXPECT_THROW(t.get({5, 0}), std::invalid_argument);  // out of range
}

TEST(TensorTest, ElementWiseDivide) {
    Tensor a({3});
    Tensor b({3});
    a.set(10.0, {0}); a.set(20.0, {1}); a.set(30.0, {2});
    b.set(2.0, {0}); b.set(5.0, {1}); b.set(10.0, {2});
    Tensor c = a / b;
    EXPECT_DOUBLE_EQ(c.get({0}), 5.0);
    EXPECT_DOUBLE_EQ(c.get({1}), 4.0);
    EXPECT_DOUBLE_EQ(c.get({2}), 3.0);
}

TEST(TensorTest, InPlaceAdd) {
    Tensor a({3});
    a.set(1.0, {0}); a.set(2.0, {1}); a.set(3.0, {2});
    Tensor b({3});
    b.fill(10.0);
    a += b;
    EXPECT_DOUBLE_EQ(a.get({0}), 11.0);
    EXPECT_DOUBLE_EQ(a.get({1}), 12.0);
    EXPECT_DOUBLE_EQ(a.get({2}), 13.0);
}

TEST(TensorTest, InPlaceSubtract) {
    Tensor a({2});
    a.fill(10.0);
    Tensor b({2});
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
    Tensor ta({2, 2}); ta.fill(1.0);
    Tensor tb({2, 2}); tb.fill(2.0);
    auto a = Node<Tensor>::make_node(ta);
    auto b = Node<Tensor>::make_node(tb);
    auto c = a + b;
    autogradient::backward(c);
    // dc/da = 1, dc/db = 1
    Tensor ones({2, 2}); ones.fill(1.0);
    EXPECT_TRUE(a->grad == ones);
    EXPECT_TRUE(b->grad == ones);
}

TEST(AutogradTensorTest, SubBackward) {
    Tensor ta({2, 2}); ta.fill(3.0);
    Tensor tb({2, 2}); tb.fill(1.0);
    auto a = Node<Tensor>::make_node(ta);
    auto b = Node<Tensor>::make_node(tb);
    auto c = a - b;
    autogradient::backward(c);
    Tensor ones({2, 2}); ones.fill(1.0);
    Tensor neg_ones({2, 2}); neg_ones.fill(-1.0);
    EXPECT_TRUE(a->grad == ones);
    EXPECT_TRUE(b->grad == neg_ones);
}

TEST(AutogradTensorTest, MulBackward) {
    // matmul: C = A * B, dC/dA = grad * B^T, dC/dB = A^T * grad
    Tensor ta({2, 3}); ta.fill(1.0);
    Tensor tb({3, 2}); tb.fill(2.0);
    auto a = Node<Tensor>::make_node(ta);
    auto b = Node<Tensor>::make_node(tb);
    auto c = a * b; // matmul -> (2,2)
    autogradient::backward(c);
    // grad is ones(2,2)
    // dC/dA = ones(2,2) * B^T = ones(2,2) * 2*ones(2,3) = 2*ones * 2 cols... let's check shapes
    EXPECT_EQ(a->grad.getShape(), (std::vector<int>{2, 3}));
    EXPECT_EQ(b->grad.getShape(), (std::vector<int>{3, 2}));
}

TEST(AutogradTensorTest, HadamardBackward) {
    Tensor ta({3}); ta.set(2.0, {0}); ta.set(3.0, {1}); ta.set(4.0, {2});
    Tensor tb({3}); tb.set(5.0, {0}); tb.set(6.0, {1}); tb.set(7.0, {2});
    auto a = Node<Tensor>::make_node(ta);
    auto b = Node<Tensor>::make_node(tb);
    auto c = Node<Tensor>::hadamard(a, b);
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
    Tensor ta({3}); ta.set(-2.0, {0}); ta.set(3.0, {1}); ta.set(-4.0, {2});
    auto a = Node<Tensor>::make_node(ta);
    auto c = abs(a);
    autogradient::backward(c);
    // d|x|/dx = sign(x)
    EXPECT_DOUBLE_EQ(a->grad.get({0}), -1.0);
    EXPECT_DOUBLE_EQ(a->grad.get({1}), 1.0);
    EXPECT_DOUBLE_EQ(a->grad.get({2}), -1.0);
}

TEST(AutogradTensorTest, MeanBackward) {
    Tensor ta({4}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(3.0, {2}); ta.set(4.0, {3});
    auto a = Node<Tensor>::make_node(ta);
    auto c = mean(a);
    autogradient::backward(c);
    // d(mean)/da_i = 1/n
    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(a->grad.get({i}), 0.25);
    }
}

TEST(AutogradTensorTest, TransposeBackward) {
    Tensor ta({2, 3}); ta.linspace(1.0, 7.0);
    auto a = Node<Tensor>::make_node(ta);
    auto c = transpose(a);
    EXPECT_EQ(c->val.getShape(), (std::vector<int>{3, 2}));
    autogradient::backward(c);
    // transpose backward is just transpose of grad
    EXPECT_EQ(a->grad.getShape(), (std::vector<int>{2, 3}));
}

TEST(AutogradTensorTest, ScalarNodeAdd) {
    Tensor ta({3}); ta.fill(2.0);
    auto a = Node<Tensor>::make_node(ta);
    auto c = a + 5.0;
    // c->val should be [7, 7, 7] with broadcasting
    autogradient::backward(c);
    // gradient of a should still flow
    EXPECT_TRUE(Tensor::has_nonzero_gradient(a->grad));
}

TEST(AutogradTensorTest, ScalarNodeSub) {
    Tensor ta({3}); ta.fill(10.0);
    auto a = Node<Tensor>::make_node(ta);
    auto c = 1.0 - a;
    autogradient::backward(c);
    // d(1-a)/da = -1
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(a->grad.get({i}), -1.0);
    }
}

TEST(AutogradTensorTest, ScalarNodeMul) {
    Tensor ta({3}); ta.fill(3.0);
    auto a = Node<Tensor>::make_node(ta);
    auto c = 2.0 * a;
    autogradient::backward(c);
    // d(2a)/da = 2
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(a->grad.get({i}), 2.0);
    }
}

TEST(AutogradTensorTest, DivBackward) {
    Tensor ta({3}); ta.fill(6.0);
    auto a = Node<Tensor>::make_node(ta);
    auto c = a / 3.0;
    autogradient::backward(c);
    // d(a/3)/da = 1/3
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(a->grad.get({i}), 1.0 / 3.0, 1e-10);
    }
}

TEST(AutogradTensorTest, LogBackward) {
    Tensor ta({3}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(4.0, {2});
    auto a = Node<Tensor>::make_node(ta);
    auto c = Node<Tensor>::log(a);
    autogradient::backward(c);
    // d(log(a))/da = 1/a
    EXPECT_NEAR(a->grad.get({0}), 1.0, 1e-10);
    EXPECT_NEAR(a->grad.get({1}), 0.5, 1e-10);
    EXPECT_NEAR(a->grad.get({2}), 0.25, 1e-10);
}

TEST(AutogradTensorTest, MaxBackward) {
    Tensor ta({3}); ta.set(1.0, {0}); ta.set(5.0, {1}); ta.set(3.0, {2});
    Tensor tb({3}); tb.set(4.0, {0}); tb.set(2.0, {1}); tb.set(3.0, {2});
    auto a = Node<Tensor>::make_node(ta);
    auto b = Node<Tensor>::make_node(tb);
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
    Tensor ta({2}); ta.fill(3.0);
    auto a = Node<Tensor>::constant(ta);
    EXPECT_DOUBLE_EQ(a->val.get({0}), 3.0);
    EXPECT_DOUBLE_EQ(a->val.get({1}), 3.0);
    // constant nodes have no backward
    EXPECT_FALSE(a->backward_fn);
}


// Loss Function Tests

TEST(LossTest, L1LossZero) {
    Tensor ta({3}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(3.0, {2});
    auto actual = Node<Tensor>::make_node(ta);
    auto pred = Node<Tensor>::make_node(ta); // same values
    auto loss = neural_network::loss_functions::l1_loss(actual, pred);
    EXPECT_NEAR(loss->val.get({0}), 0.0, 1e-10);
}

TEST(LossTest, L1LossValue) {
    Tensor ta({4}); ta.fill(0.0);
    Tensor tp({4}); tp.set(1.0, {0}); tp.set(-1.0, {1}); tp.set(2.0, {2}); tp.set(-2.0, {3});
    auto actual = Node<Tensor>::make_node(ta);
    auto pred = Node<Tensor>::make_node(tp);
    auto loss = neural_network::loss_functions::l1_loss(actual, pred);
    // |1| + |-1| + |2| + |-2| = 6, mean = 1.5
    EXPECT_NEAR(loss->val.get({0}), 1.5, 1e-10);
}

TEST(LossTest, L2LossZero) {
    Tensor ta({3}); ta.set(1.0, {0}); ta.set(2.0, {1}); ta.set(3.0, {2});
    auto actual = Node<Tensor>::make_node(ta);
    auto pred = Node<Tensor>::make_node(ta);
    auto loss = neural_network::loss_functions::l2_loss(actual, pred);
    EXPECT_NEAR(loss->val.get({0}), 0.0, 1e-10);
}

TEST(LossTest, L2LossValue) {
    Tensor ta({3}); ta.fill(0.0);
    Tensor tp({3}); tp.set(1.0, {0}); tp.set(2.0, {1}); tp.set(3.0, {2});
    auto actual = Node<Tensor>::make_node(ta);
    auto pred = Node<Tensor>::make_node(tp);
    auto loss = neural_network::loss_functions::l2_loss(actual, pred);
    // (1+4+9)/3 = 14/3
    EXPECT_NEAR(loss->val.get({0}), 14.0 / 3.0, 1e-10);
}

TEST(LossTest, ShapeMismatchThrows) {
    Tensor ta({2}); ta.fill(1.0);
    Tensor tp({3}); tp.fill(1.0);
    auto a = Node<Tensor>::make_node(ta);
    auto p = Node<Tensor>::make_node(tp);
    EXPECT_THROW(neural_network::loss_functions::l1_loss(a, p), std::runtime_error);
    EXPECT_THROW(neural_network::loss_functions::l2_loss(a, p), std::runtime_error);
}


// Linear Layer Tests

TEST(LinearTest, OutputShape) {
    neural_network::Linear layer(5, 10);
    Tensor input({1, 5});
    input.fill(1.0);
    auto x = Node<Tensor>::make_node(input);
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
    Tensor input({1, 3});
    input.fill(1.0);
    auto x = Node<Tensor>::make_node(input);
    auto out = layer(x);
    autogradient::backward(out);
    auto params = layer.parameters();
    EXPECT_TRUE(Tensor::has_nonzero_gradient(params[0]->grad)); // weight has grad
    EXPECT_TRUE(Tensor::has_nonzero_gradient(params[1]->grad)); // bias has grad
}


// ReLU Tests

TEST(ReLUTest, ForwardPositive) {
    neural_network::ReLU relu;
    Tensor input({4});
    input.set(-2.0, {0}); input.set(0.0, {1}); input.set(3.0, {2}); input.set(-1.0, {3});
    auto x = Node<Tensor>::make_node(input);
    auto out = relu(x);
    EXPECT_DOUBLE_EQ(out->val.get({0}), 0.0);
    EXPECT_DOUBLE_EQ(out->val.get({1}), 0.0);
    EXPECT_DOUBLE_EQ(out->val.get({2}), 3.0);
    EXPECT_DOUBLE_EQ(out->val.get({3}), 0.0);
}


// SGD Optimizer Tests

TEST(SGDTest, StepUpdatesParams) {
    Tensor tw({2}); tw.fill(5.0);
    auto w = Node<Tensor>::make_node(tw);
    w->grad = Tensor({2}); w->grad.fill(1.0); // manually set grad

    neural_network::optimizers::SGD sgd({w}, 0.1);
    sgd.step();
    // w = w - lr * grad = 5 - 0.1*1 = 4.9
    EXPECT_NEAR(w->val.get({0}), 4.9, 1e-10);
    EXPECT_NEAR(w->val.get({1}), 4.9, 1e-10);
}

TEST(SGDTest, ZeroGrad) {
    Tensor tw({3}); tw.fill(1.0);
    auto w = Node<Tensor>::make_node(tw);
    w->grad = Tensor({3}); w->grad.fill(5.0);

    neural_network::optimizers::SGD sgd({w}, 0.01);
    sgd.zero_grad();
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(w->grad.get({i}), 0.0);
    }
}


// main to run all the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
