//
// Created by richard on 6/22/24.
//

#ifndef OPENXAE_TORCH_OPTIMIZATION_H
#define OPENXAE_TORCH_OPTIMIZATION_H

#include "Graph.h"

#include <torch/script.h>

namespace pnnx {

ParameterVar CreateParameterFromTorchNode(const torch::jit::Node* value_node);

ParameterVar CreateParameterFromTorchValue(const torch::jit::Value* value);

std::shared_ptr<torch::jit::Graph> OptimizeTorchScript(torch::jit::Module& mod);

}

#endif//OPENXAE_TORCH_OPTIMIZATION_H
