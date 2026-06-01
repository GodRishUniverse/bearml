// #include "tensor/Tensor.h"
// #include "autograd/autogradient.h"
// #include "activation_functions/modules.h"
// #include "model_construct/model_construct.h"
// #include "loss_functions/loss.h"
// #include "optimizers/optimizers.h"
// #include "devices/device_type.h"
#include "simplenet.h"
#include <iostream>
#include <vector>



using namespace std;

class Model : public simplenet::neural_network::Model_Construct{
    public:
    simplenet::neural_network::Linear layer1;
    simplenet::neural_network::Tanh nonlinearity;
    simplenet::neural_network::Linear layer2;

    Model(int in_shape, int out_shape, simplenet::Device dev = simplenet::Device(simplenet::DeviceType::CPU, 0)):  layer1(in_shape, out_shape, "Xavier", dev), nonlinearity(42, dev), layer2(out_shape, out_shape, "Xavier", dev) {

    }

    std::shared_ptr<simplenet::Node<simplenet::TensorD>> forward(std::vector<simplenet::TensorD> inputs) override {
        // assuming only x is passed in
        auto x = simplenet::Node<simplenet::TensorD>::make_node(inputs[0]);

        auto f1 = this->layer1(x);
        auto f2 = this->nonlinearity(f1);
        return this->layer2(f2);
    }

    std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> parameters() override {
        std::vector<std::shared_ptr<simplenet::Node<simplenet::TensorD>>> params;

        // collect parameters from layer1
        auto l1_params = layer1.parameters();
        params.insert(params.end(), l1_params.begin(), l1_params.end());

        // ReLU has no parameters - although an override is there in the class

        // collect parameters from layer2
        auto l2_params = layer2.parameters();
        params.insert(params.end(), l2_params.begin(), l2_params.end());

        return params;
    }

};

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


    // simplenet::Device dev =         simplenet::Device(simplenet::DeviceType::CUDA,  0);

    // veca.to_(dev);
    // cout << veca<< endl;


    // simplenet::Tensor vecb ({3});
    // veca.set(201.0, {1, 0});
    // vecb.set(2.0, {1});

    // cout << veca * vecb << endl;

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

    // simplenet::Tensor a ({1,1,5,5});
    // simplenet::Tensor b ({1,2,5,5});

    // a.linspace(1,97);
    // b.linspace(1,10);

    // cout << a * b << endl;


    // shared_ptr<simplenet::Node<simplenet::Tensor>> x = simplenet::Node<simplenet::Tensor>::make_node(a); // calling static function
    // shared_ptr<simplenet::Node<simplenet::Tensor>> y = simplenet::Node<simplenet::Tensor>::make_node(b); // calling static function

    // auto z = x*y + x;

    // cout << simplenet::autogradient::backward(z) << endl;
    // cout << x->grad << endl;
    // cout << y->grad << endl;

    // simplenet::Tensor c({3,2,4});
    // c.set(1.0, {0,1,1});
    // cout << c << endl;
    // c =  c.flatten(1, -1, true );
    // cout << c << endl;

    // simplenet::Tensor d({2,2,3,4,2});
    // d.linspace(1,97);
    // cout << d << endl;
    // cout << d.getShape().size() << endl;

    // cout << d.sum(4,false) << endl;


    // cout << "FLATTENING IN PLACE" << endl;
    // d.flatten_inplace(1,2,false);
    // cout << d << endl;
    //



    // simplenet::Tensor test_tensor_for_reduce({7,2,2,3,4,2});
    // vector<int> test_temp = {2,2,4,2};
    // cout << simplenet::linear_algebra::reduce(test_tensor_for_reduce,test_temp  )<< endl;

    // simplenet::Tensor tester({1,2,4,5});
    // tester.linspace(1,10);


    // shared_ptr<simplenet::Node<simplenet::Tensor>> node = simplenet::Node<simplenet::Tensor>::make_node(tester);

    // simplenet::neural_network::Linear layer1 = simplenet::neural_network::Linear(5, 10);
    // cout << "after linear layer" << endl;

    // auto vvv = layer1(node);


    // simplenet::neural_network::ReLU layer_relu = simplenet::neural_network::ReLU();
    // cout << "after relu layer" << endl;

    // auto vvvv= layer_relu(vvv);

    // simplenet::autogradient::backward(vvvv);
    // cout << vvv->val << endl;

    simplenet::Device dev = simplenet::Device(simplenet::DeviceType::CUDA,  0);

    Model testmodel = Model(2, 5, dev);

    simplenet::TensorD tester2({1,2});
    tester2.linspace(1,2);
    tester2.to_(dev);

    // so PyTorch uses its custom random number generator for initialization and so our initialization does not match and so for testing this what can be done is we basically get pytorch weights and see if the numbers match
    auto pred = testmodel.forward({tester2});
    pred->val.to_(dev);

    // cout << pred->val << endl;
    // cout << "Pred grad: " << pred->grad << endl;


    simplenet::TensorD actual({1,5});
    actual.linspace(1,5);
    actual.to_(dev);
    simplenet::neural_network::optimizers::SGD optim(testmodel.parameters(), 0.1);

    // Sample - works gets closer to the ideal values
    for (int i =0;i <100; i++){
        optim.zero_grad();
        auto ac = simplenet::Node<simplenet::TensorD>::make_node(actual);

        auto loss = simplenet::neural_network::loss_functions::l1_loss(ac, pred);
        simplenet::autogradient::backward(loss);

        optim.step();

        cout << "Loss value: " << loss->val << endl;

        pred = testmodel.forward({tester2});
        cout << "Pred value: " << pred->val << endl;
    }
    cout << "Pred grad: " << pred->grad << endl;



    // simplenet::Tensor a ({5});
    // a.linspace(1,5);

    // cout << a << endl;
    // simplenet::Tensor b ({5,3});
    // b.fill(2.0);

    // a.to_(dev);
    // b.to_(dev);

    // auto c = a * b;
    // cout << c << endl;

     simplenet::TensorD mat_inv ({5,5});
     mat_inv.set(1.0, {0,0});
     mat_inv.set(1.0, {1,1});
     mat_inv.set(1.0, {2,2});
     mat_inv.set(1.0, {3,3});
     mat_inv.set(-1.0, {4,4});

     mat_inv.to_(dev);


     cout << "CONCAT " << endl;
     cout << simplenet::TensorD::concat({mat_inv, mat_inv}, 1) << endl;

     cout << "PADDING " << endl;
     auto padded = simplenet::neural_network::padding<double>(mat_inv, 1,Padding_Op_Code::PAD_CONSTANT, 2.0 );
     cout << padded << endl;

     // simplenet::neural_network::LeakyReLU leaky_relu(0.1);
     // auto leaky_relu_node = simplenet::Node<simplenet::Tensor>::make_node(mat_inv);
     // auto leaky_relu_out = leaky_relu.forward(leaky_relu_node);
     // cout << leaky_relu_out->val << endl;
     // mat_inv.fill(1.0);
     // simplenet::Tensor mat_inv_2 ({5,2});
     //

     padded.to_(simplenet::Device::cpu());
     simplenet::TensorD::setPrintPrecision(3);
     simplenet::TensorD cont = simplenet::TensorD::contiguous( padded.slice("1, 1:5:2"));
     cout << cont << endl;


     // cout << cont({0,1}) << endl;
     //
     auto tens = simplenet::TensorD::ones({1,1,4,4}, dev);
     tens.linspace(1.0, 17.0);

     cout << "TENS " << endl;
     cout << tens << endl;


     cout << "TENS float" << endl;
     cout <<  tens.change_dtype<int>() << endl;

     // ---------------- transpose / permute view checks ----------------
     // build a 2x3 with values 1..6 on cpu so we can read values back
     simplenet::TensorD mat ({2,3});
     mat.linspace(1.0, 6.0); // [[1,2,3],[4,5,6]]
     cout << "MAT (2x3)" << endl;
     cout << mat << endl;

     // transpose: should be O(1) view with shape (3,2), strides swapped, layout STRIDED
     simplenet::TensorD matT = mat.transpose();
     cout << "MAT.transpose() shape=" << matT.getShape()[0] << "x" << matT.getShape()[1]
          << "  strides=[" << matT.getStrides()[0] << "," << matT.getStrides()[1] << "]"
          << "  layout=" << (matT.layout() == simplenet::utils::Layout::ROW_MAJOR ? "ROW"
                            : matT.layout() == simplenet::utils::Layout::COL_MAJOR ? "COL" : "STRIDED")
          << "  shares_data=" << ((matT.getStridesColMajor().size() == mat.getShape().size()) ? "yes" : "no") << endl;
     cout << matT << endl;  // print uses strides; expect [[1,4],[2,5],[3,6]]

     // densifying the view should give the same values in row-major layout
     simplenet::TensorD matT_dense = matT.contiguous();
     cout << "MAT.transpose().contiguous() layout="
          << (matT_dense.layout() == simplenet::utils::Layout::ROW_MAJOR ? "ROW"
             : matT_dense.layout() == simplenet::utils::Layout::COL_MAJOR ? "COL" : "STRIDED") << endl;
     cout << matT_dense << endl;

     // permute on a 3D tensor: (2,3,4) permuted with order {2,0,1} -> shape (4,2,3)
     simplenet::TensorD cube ({2,3,4});
     cube.linspace(1.0, 24.0);
     simplenet::TensorD cubeP = cube.permute({2,0,1});
     cout << "CUBE.permute({2,0,1}) shape=" << cubeP.getShape()[0] << "x"
          << cubeP.getShape()[1] << "x" << cubeP.getShape()[2]
          << "  layout=" << (cubeP.layout() == simplenet::utils::Layout::ROW_MAJOR ? "ROW"
                            : cubeP.layout() == simplenet::utils::Layout::COL_MAJOR ? "COL" : "STRIDED") << endl;
     cout << cubeP << endl;

     // cout << "IM2COL " << endl;
     // cout << simplenet::linear_algebra::im2col_2d(tens, 3, 1, 0, 1) << endl;





     // auto concat_out = simplenet::Tensor::concat({mat_inv, mat_inv_2}, 1);
     // cout << concat_out << endl;


     // // we need to use simplenet ops (shouldn't these be in linalg?)
     // cout << simplenet::Tensor::tan(mat_inv) << endl;
     simplenet::TensorD t({5});
     t.linspace(0.0, 8.0);
     cout << t << endl;


}
