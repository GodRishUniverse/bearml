// #include "Matrix.cpp"
// #include "Tensor.cpp"
#include "autogradient.cpp"

#include <iostream>
using namespace std;

int main() {

    // Matrix m1(4,4);
    // cout << m1 << endl;
    // Matrix m2(4,4);
    // cout << m2 << endl;

    // simplenet::Matrix<int> m3 {{1, 2}, {5, 6}, {9, 10}, {13, 14}};

    // cout << m3 << endl;

    // simplenet::Matrix<int> m4({{1,4,5,6}, {7,8,9,10}});

    // simplenet::Matrix<int> m6({{1,4,5,6}, {7,8,9,10}});

    // cout << m4 << endl;

    // simplenet::Matrix<int> m5 = m3 * m4;

    // cout << m5 << endl;

    // vector<int> size; 
    // size.push_back(2);
    // size.push_back(3);
    // size.push_back(2);
    // simplenet::Tensor t{size};
    // cout << t << endl;
    // simplenet::Tensor t2{size};
    // cout << t2 << endl;
    // simplenet::Tensor t3{size};
    // cout << t3 << endl;

    // t.stack({t2, t3});
    // t.printShape();

    // cout << t << endl;

    // cout << m4 + m6 << endl;


    // for (size_t s = 0; s <t.getShape().size(); s++){
    //     cout << t.getShape()[s] <<endl;
    // }

    // vector<int> reshape_shape {1,1,6,6};

    // t.reshape(reshape_shape);
    // cout << "AFTER RESHAPE" <<endl;
    // for (size_t s = 0; s <t.getShape().size(); s++){
    //     cout << t.getShape()[s] <<endl;
    // }
    // cout << t<< endl;
    // t.squeeze(3);
    // cout << t <<endl;

    // t.flatten<void>();
    // cout << t << endl;

    // vector<int> tempS {2,3,4};
    // simplenet::Tensor newT_A {tempS};
    // simplenet::Tensor newT_B {tempS};
    // newT_A.set(10, {1,0,0});
    // newT_B.set(10, {1,0,0});

    // cout << newT_A <<endl;
    // cout << newT_B <<endl;

    // cout << newT_A * newT_B << endl; // hadamard product works

    double x_val=  4.0;
    double y_val = 2.0;

    shared_ptr<simplenet::Node<double>> x = simplenet::Node<double>::make_node(x_val); // calling static function
    shared_ptr<simplenet::Node<double>> y = simplenet::Node<double>::make_node(y_val); // calling static function

    auto z = x*y + x;
    // dz/dx = y+1 = 2+1 =3
    // dz/dy= x =4


    cout << simplenet::autogradient::backward(z) << endl;
    cout << x->grad << endl;
    cout << y->grad << endl;


}
