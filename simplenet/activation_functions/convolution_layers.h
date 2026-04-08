#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"
#include "modules.h"
#include "../operators/padding_ops.h"


#ifndef ACTIVATION_FUNCTIONS_CONVOLUTION
#define ACTIVATION_FUNCTIONS_CONVOLUTION

//TODO: write convolution layers here - Conv1D, Conv2D, Conv3d, ConvTranspose1D, ConvTranpose2D, ConvTranspose3D,
namespace simplenet {

    namespace neural_network {

        simplenet::Tensor padding(const simplenet::Tensor& input, int pad_amount, Padding_Op_Code padding_mode = Padding_Op_Code::PAD_ZERO){
            // now we need to change the shape
            std::vector<int> output_shape = input.getShape();
            std::vector<int> input_shape = input.getShape();
            // check if the input is a matrix or not at least for pad_amount - cause i dont think 1D padding works
            if (input_shape.size() < 2) {
                throw std::invalid_argument("Padding requires at least 2D input");
            }
            // add pad_amount to the shape
            output_shape[output_shape.size() - 2] += 2 * pad_amount; // add 2*pad_amount rows
            output_shape[output_shape.size() - 1] += 2 * pad_amount; // add 2*pad_amount columns

            simplenet::Tensor output(output_shape, input.getDevice());

            long long int batch_size = input.sizeOfTensor() / (input_shape[input_shape.size() - 2] * input_shape[input_shape.size() - 1]);

            switch (padding_mode) {
                case Padding_Op_Code::PAD_ZERO:
                    // NOTE: aparrantly variable declaration inside the switch statement needs case (cond): {} rather than case (cond): only
                    // now if we have padding_mode as zeros then we can just need to copy the input values appropriately
                    // n*m matrix becomes (n+2p) * (m+2p) matrix
                    if (input.getDevice().is_cpu()) {
                        for (int batch = 0; batch < batch_size; batch++) {
                            for (int r = 0; r < input_shape[input_shape.size() - 2]; r++) {
                                for (int c = 0; c < input_shape[input_shape.size() - 1]; c++) {
                                    output.data[batch * output_shape[output_shape.size() - 1] * output_shape[output_shape.size() - 2] + (r+pad_amount)*output_shape[output_shape.size() - 1] + (c+pad_amount)] = input.data[batch * input_shape[input_shape.size() - 2] * input_shape[input_shape.size() - 1] + r * input_shape[input_shape.size() - 1] + c];
                                }
                            }
                        }
                    } else {
                        // CUDA
                        simplenet::cuda::launch_padd_with_zeroes<double>(input.data, output.data, batch_size, input_shape[input_shape.size() - 2], input_shape[input_shape.size() - 1], pad_amount);
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported padding mode");
            }
            return output;
        }

        // TODO: need to figure out how interleaving works for padding before I can implement convolution layers (and pooling)
        // template copied from pytorch doc - https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        // class Conv1D : public Module {
        // private:
        //     int in_channels_;
        //     int out_channels_;
        //     int kernel_size_;
        //     int stride_;
        //     int padding_;
        //     int dilation_;
        //     int groups_;
        //     bool bias_;
        //     std::string padding_mode_;


        // public:
        //     Conv1D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation =1, int groups = 1, bool bias = true, std::string padding_mode = "zeros", int seed = 42, simplenet::Device dev = simplenet::Device::cpu())
        //         : Module(seed, dev), in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation), groups_(groups), bias_(bias), padding_mode_(padding_mode) {

        //     }
        // };

    }
}
#endif
