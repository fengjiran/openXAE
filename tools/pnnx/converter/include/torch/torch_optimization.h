//
// Created by richard on 6/22/24.
//

#ifndef OPENXAE_TORCH_OPTIMIZATION_H
#define OPENXAE_TORCH_OPTIMIZATION_H

#include "Graph.h"
#include "pass_level0.h"

#include <torch/script.h>
#include <torch/csrc/api/include/torch/version.h>

namespace pnnx {

std::shared_ptr<torch::jit::Graph> OptimizeTorchScript(torch::jit::Module& mod,
                                                       const std::vector<at::Tensor>& inputTensors,
                                                       const std::vector<at::Tensor>& inputTensors2,
                                                       const std::vector<std::string>& moduleOperators,
                                                       const std::string& ptPath,
                                                       const std::string& device,
                                                       std::set<std::string>& foldableConstants,
                                                       const std::string& foldableConstantsZippath);

Parameter CreateParameterFromTorchNode(const torch::jit::Node* node);

Parameter CreateParameterFromTorchValue(const torch::jit::Value* value);

}

#endif//OPENXAE_TORCH_OPTIMIZATION_H
