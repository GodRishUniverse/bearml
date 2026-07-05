#pragma once

// This is the library for the BearML framework. - all files are included in this header file.
// Only include this header file to use the BearML framework.
// BearML is a neural network framework written in C++ with CUDA support.

#include "tensor/Tensor.h"
#include "autograd/autogradient.h"
#include "activation_functions/modules.h" // has the abstract Modules class and the Linear and ReLU modules
#include "activation_functions/activation_functions.h" // major activation functions
#include "activation_functions/pooling_layers.h" // pooling layers
#include "activation_functions/convolution_layers.h" // convolution layers
#include "model_construct/model_construct.h"
#include "loss_functions/loss.h"
#include "optimizers/optimizers.h"
#include "devices/device_type.h"
#include "operators/ops.h"
#include "operators/padding_ops.h"
#include "dataloader/dataloader.h"
#include "datasets/datasets.h"
// #include "cuda/includes/"
