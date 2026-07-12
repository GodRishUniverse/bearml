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

        template <typename T>
        bearml::Tensor<T> padding(const bearml::Tensor<T>& input, int pad_amount, Padding_Op_Code padding_mode, T constant_value){
            // now we need to change the shape
            std::vector<int> output_shape = input.getShape();
            std::vector<int> input_shape = input.getShape();
            // check if the input is a matrix or not at least for pad_amount - cause i dont think 1D padding works
            if (input_shape.size() < 2) {
                throw std::invalid_argument("Padding requires at least 2D input");
            }
            // add pad_amount to the shape
            output_shape[output_shape.size() - 2] += 2 * pad_amount; // add 2*pad_amount rows (or subtract if pad_amount is negative)
            output_shape[output_shape.size() - 1] += 2 * pad_amount; // add 2*pad_amount columns (or subtract if pad_amount is negative)

            bearml::Tensor<T> output(output_shape, input.getDevice());

            long long int batch_size = input.sizeOfTensor() / (input_shape[input_shape.size() - 2] * input_shape[input_shape.size() - 1]);

            int in_rows  = input_shape[input_shape.size() - 2];
            int in_cols  = input_shape[input_shape.size() - 1];
            int out_rows = output_shape[output_shape.size() - 2];
            int out_cols = output_shape[output_shape.size() - 1];

            // Only the region where input and (possibly cropped) output overlap gets copied;
            // walking the full input range and offsetting by pad_amount only works when
            // pad_amount >= 0 -- for pad_amount < 0 (cropping) that walks off both ends of
            // output's allocation. Clamping to the overlap keeps every write in-bounds either way.
            // Computed once here (rather than separately per-backend) since CPU and CUDA
            // dispatch below both need the exact same overlap window.
            int r_start = std::max(0, -pad_amount);
            int r_end   = std::min(in_rows, out_rows - pad_amount);
            int c_start = std::max(0, -pad_amount);
            int c_end   = std::min(in_cols, out_cols - pad_amount);

            switch (padding_mode) {
                case Padding_Op_Code::PAD_CONSTANT:
                    // NOTE: aparrantly variable declaration inside the switch statement needs case (cond): {} rather than case (cond): only
                    // now if we have padding_mode as zeros then we can just need to copy the input values appropriately
                    // n*m matrix becomes (n+2p) * (m+2p) matrix
                    if (input.getDevice().is_cpu()) {
                        // first fill the output tensor with the constant value
                        for (int i = 0; i < output.sizeOfTensor(); i++) {
                            output.data[i] = constant_value;
                        }

                        for (int batch = 0; batch < batch_size; batch++) {
                            for (int r = r_start; r < r_end; r++) {
                                for (int c = c_start; c < c_end; c++) {
                                    output.data[batch * out_cols * out_rows + (r+pad_amount)*out_cols + (c+pad_amount)] = input.data[batch * in_rows * in_cols + r * in_cols + c];
                                }
                            }
                        }
                    } else {
                        // CUDA
                        bearml::cuda::launch_padd_with_constant<cuda_type_trait_t<T>>(cuda_ptr(input.data), cuda_ptr(output.data), batch_size, in_rows, in_cols, out_rows, out_cols, pad_amount, r_start, r_end, c_start, c_end, cuda_val(constant_value));
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported padding mode");
            }
            return output;
        }

        // we have to support 2 types of convolution operations depending on the input shape (im2col for kernelsize<= 7 and FFT for larger kernels)
        // Wow we have measure theory and lebesgue inttegrals to get the derivative oof a convolution!!!
        // Interesting - https://math.stackexchange.com/questions/177239/derivative-of-convolution
        // I haven't delved into this theory yet so will use this as a baseline
        // TODO: do more research on the derivative of convolution

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
            int in_channels_;
            int out_channels_;
            int kernel_size_;
            int stride_;
            int padding_;
            int dilation_;
            int groups_;
            bool bias_;
            Padding_Op_Code padding_mode_;
            int constant_pad_;

        public:
            Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation =1, int groups = 1, bool bias = true, Padding_Op_Code padding_mode = Padding_Op_Code::PAD_CONSTANT, int seed = 42, bearml::Device dev = bearml::Device::cpu(), int constant_pad = 0)
                : Module<T>(seed, dev), in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation), groups_(groups), bias_(bias), padding_mode_(padding_mode), constant_pad_(constant_pad) {

            }

            std::shared_ptr<bearml::Node<T>> forward(std::shared_ptr<bearml::Node<T>> x) override {
                // implement im2col and then use GEMM - rather than using FFT
            }
        };

    }
}
#endif
