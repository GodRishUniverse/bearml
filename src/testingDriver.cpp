// #include "Matrix.cpp"
#include "Tensor.cpp"
#include "Tensor.h"
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
    //     vector<int> tempS_1 {2,3,4,7};
    // simplenet::Tensor newT_A {tempS};
    // simplenet::Tensor newT_B {tempS};
    // simplenet::Tensor newT_C {tempS_1};
    // // newT_A.set(10.0, {1,0,0});

    // // newT_B.set(10.0000000000001, {1,0,0});

    // cout << (newT_A == newT_C) << endl;



    // cout << newT_A <<endl;
    // cout << newT_C <<endl;

    // cout << newT_A * newT_B << endl; // hadamard product works

    // double x_val=  4.0;
    // double y_val = 2.0;

    // shared_ptr<simplenet::Node<double>> x = simplenet::Node<double>::make_node(x_val); // calling static function
    // shared_ptr<simplenet::Node<double>> y = simplenet::Node<double>::make_node(y_val); // calling static function

    // auto z = x*y + x;
    // // dz/dx = y+1 = 2+1 =3
    // // dz/dy= x =4


    // cout << simplenet::autogradient::backward(z) << endl;
    // cout << x->grad << endl;
    // cout << y->grad << endl;

    // simplenet::Tensor a({2, 1, 3});
    // simplenet::Tensor b({4, 3});


    // // Fill with test data
    // for (int i = 0; i < 6; ++i) a.data[i] = i + 1;      // [1,2,3,4,5,6]
    // cout << a << endl;
    // for (int i = 0; i < 12; ++i) b.data[i] = (i + 1) * 10; // [10,20,30,40,50,60,70,80,90,100,110,120]
    // cout << b << endl;

    // simplenet::Tensor result = a + b;

    // cout << result << endl;
    // simplenet::Tensor c ({1});
    // c.set(1.5, {0});

    // // testing some scalar mul operations
    // cout << 1.1*result <<endl;
    // cout << result*1.1 <<endl;
    // cout << c*result << endl;
    // cout << result*c << endl;
    // cout << c*c << endl;

    // cout << result << endl;
    // result*=c;
    // cout << result << endl;

    // result*=7.0;
    // cout << result << endl;


    // Let us test the transpose function

    // cout <<"Testing TRANSPOSE (NOT in-place)-> \nHERE WE HAVE B Tensor: \n" << b <<endl;
    // cout << "TRANSPOSED to:\n" << endl;
    // simplenet::Tensor b_transpose =simplenet::Tensor::transpose(b, 0, 1);
    // cout << b_transpose << endl;
    // b_transpose.set(10.0, {1,3});
    // cout <<"VALUE CHANGED IN TRANSPOSE is:\n" <<endl;
    // cout << b_transpose << endl;
    // cout << b << endl;


    // cout <<"Testing TRANSPOSE (NOT in-place)-> \nHERE WE HAVE A Tensor: \n" << a <<endl;
    // cout << "TRANSPOSED to:\n" << endl;
    // simplenet::Tensor a_transpose =simplenet::Tensor::transpose(a, 0, 1);
    // cout << a_transpose << endl;
    // a_transpose.set(10.0, {1,2,1});
    // cout <<"VALUE CHANGED IN TRANSPOSE is:\n" <<endl;
    // cout << a_transpose << endl;
    // cout << a << endl;
    //

    simplenet::Tensor veca ({3,3});
    simplenet::Tensor vecb ({3});
    veca.set(201.0, {1, 0});
    vecb.set(2.0, {1});

    cout << veca * vecb << endl;


}
