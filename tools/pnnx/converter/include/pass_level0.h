//
// Created by richard on 6/19/24.
//

#ifndef OPENXAE_PASS_LEVEL0_H
#define OPENXAE_PASS_LEVEL0_H

#include "Graph.h"
#include "inline_block.h"

#include <torch/script.h>

namespace pnnx {

void pass_level0(const torch::jit::Module& mod,
                 std::shared_ptr<torch::jit::Graph>& g,
                 const std::vector<at::Tensor>& input_tensors,
                 const std::vector<at::Tensor>& input_tensor2,
                 const std::vector<std::string>& module_operators,
                 const std::string& ptpath,
                 const std::string& device,
                 std::set<std::string>& foldable_constants,
                 const std::string& foldable_constants_zippath);

}

#endif//OPENXAE_PASS_LEVEL0_H
