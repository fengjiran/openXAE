//
// Created by richard on 7/3/24.
//

#ifndef OPENXAE_SHAPE_INFERENCE_H
#define OPENXAE_SHAPE_INFERENCE_H

#include <map>
#include <torch/script.h>

namespace pnnx {

void ShapeInference(const torch::jit::Module& mod,
                    std::shared_ptr<torch::jit::Graph>& graph,
                    const std::vector<at::Tensor>& inputTensors,
                    const std::vector<at::Tensor>& inputTensors2,
                    const std::vector<std::string>& moduleOperators,
                    const std::string& ptpath,
                    const std::string& device,
                    std::set<std::string>& foldableConstants,
                    const std::string& foldableConstantsZippath);

}

#endif//OPENXAE_SHAPE_INFERENCE_H
