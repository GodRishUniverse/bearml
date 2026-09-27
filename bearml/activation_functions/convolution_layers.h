#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <algorithm>

#include "autograd/autogradient.h"
#include "tensor/Tensor.h"
#include "modules.h"
#include "../operators/padding_ops.h"
#include <fftw3.h> // will allow us to use FFT for convolution


#ifndef ACTIVATION_FUNCTIONS_CONVOLUTION
#define ACTIVATION_FUNCTIONS_CONVOLUTION

//TODO: write convolution layers here - Conv1D, Conv2D, Conv3d, ConvTranspose1D, ConvTranpose2D, ConvTranspose3D,
namespace bearml {

    namespace neural_network {


        // we have to support 2 types of convolution operations depending on the input shape (im2col for kernelsize<= 7 and FFT for larger kernels)
        // Wow we have measure theory and lebesgue inttegrals to get the derivative oof a convolution!!!
        // Interesting - https://math.stackexchange.com/questions/177239/derivative-of-convolution
        // I haven't delved into this theory yet so will use this as a baseline
        // TODO: do more research on the derivative of convolution
        //
        // Intuition - https://betterexplained.com/articles/intuitive-convolution/https://betterexplained.com/articles/intuitive-convolution/

        // derivative of conv(f,g) is conv(f',g) where f and g are vectors (1d)
        template <typename T>
        Tensor<T> conv1d(...) {
            // TODO
        }


        template <typename T>
        Tensor<T> conv2d(...) {
            // TODO
        }


        template <typename T>
        Tensor<T> conv3d(...) {
            // TODO
        }

        // TODO: need to figure out how I can implement convolution layers (and pooling)
        // TODO: we have to support 2 types of convolution operations depending on the input shape (im2col for kernelsize<= 7 and FFT for larger kernels)
        // Will probably create a separate function for each type of convolution operation and then call it based on the input shape
        // TODO: backward pass for bothh will be different so will need to implement separately and understand how to identify which backward pass to use
        // template copied from pytorch doc - https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        template <typename T = bearml::Tensorf>
        class Conv2D : public Module<T> {
        private:
            int in_channels;
            int out_channels;
            int kernel_size;
            int stride;
            int padding;
            int dilation;
            int groups;
            bool bias;
            Padding_Op_Code padding_mode;
            int constant_pad;

        public:
            Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation =1, int groups = 1, bool bias = true, Padding_Op_Code padding_mode = Padding_Op_Code::PAD_CONSTANT, int seed = 42, bearml::Device dev = bearml::Device::cpu(), int constant_pad = 0)
                : Module<T>(seed, dev), in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), groups(groups), bias(bias), padding_mode(padding_mode), constant_pad(constant_pad) {

            }

            T forward(T& x) override {
                this->check_device(x);
                // implement im2col and then use GEMM - rather than using FFT
                throw std::logic_error("Conv2D::forward is not implemented yet");
            }

            T operator()(T& x) override {
                return this->forward(x);
            }

            std::vector<std::shared_ptr<T>> parameters() override {
                return {};
            }
        };

    }
}
#endif
