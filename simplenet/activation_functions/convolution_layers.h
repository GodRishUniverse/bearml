#pragma once
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"
#include "modules.h"

#ifndef ACTIVATION_FUNCTIONS_CONVOLUTION
#define ACTIVATION_FUNCTIONS_CONVOLUTION

//TODO: write convolution layers here - Conv1D, Conv2D, Conv3d, ConvTranspose1D, ConvTranpose2D, ConvTranspose3D,
namespace simplenet {

    namespace neural_network {

        // template copied from pytorch doc - https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        class Conv1D : public Module {
        private:
            int in_channels_;
            int out_channels_;
            int kernel_size_;
            int stride_;
            int padding_;
            int dilation_;
            int groups_;
            bool bias_;
            std::string padding_mode_;


        public:
            Conv1D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation =1, int groups = 1, bool bias = true, std::string padding_mode = "zeros", int seed = 42, simplenet::Device dev = simplenet::Device::cpu())
                : Module(seed, dev), in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation), groups_(groups), bias_(bias), padding_mode_(padding_mode) {

            }
        };

    }
}
#endif
