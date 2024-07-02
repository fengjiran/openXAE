//
// Created by richard on 6/22/24.
//

#ifndef OPENXAE_TORCH_OPTIMIZATION_H
#define OPENXAE_TORCH_OPTIMIZATION_H

#include "Graph.h"
#include "constant_unpooling.h"
#include "inline_block.h"
#include "reset_device.h"

#include <torch/script.h>
#include <torch/csrc/api/include/torch/version.h>

namespace pnnx {

std::shared_ptr<torch::jit::Graph> OptimizeTorchScript(torch::jit::Module& mod, const std::string& device);

ParameterVar CreateParameterFromTorchNode(const torch::jit::Node* value_node);

ParameterVar CreateParameterFromTorchValue(const torch::jit::Value* value);

}

#endif//OPENXAE_TORCH_OPTIMIZATION_H
