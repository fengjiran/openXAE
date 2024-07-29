//
// Created by richard on 7/10/24.
//

#ifndef OPENXAE_PASS_LEVEL0_H
#define OPENXAE_PASS_LEVEL0_H

#include <map>
#include <torch/script.h>

namespace pnnx {

void inline_block(std::shared_ptr<torch::jit::Graph>& graph,
                  const std::vector<std::string>& moduleOperators);

void Inline(std::shared_ptr<torch::jit::Graph>& graph,
            const std::vector<std::string>& moduleOps);

void ConstantUnpooling(std::shared_ptr<torch::jit::Graph>& graph);

void ConvertHalf2Float(torch::jit::Module& mod);

void ResetDevice(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device);

void FlattenInput(std::shared_ptr<torch::jit::Graph>& graph);

void ShapeInference(const torch::jit::Module& mod,
                    std::shared_ptr<torch::jit::Graph>& graph,
                    const std::vector<at::Tensor>& inputTensors,
                    const std::vector<at::Tensor>& inputTensors2,
                    const std::vector<std::string>& moduleOperators,
                    const std::string& ptpath,
                    const std::string& device,
                    std::set<std::string>& foldableConstants,
                    const std::string& foldableConstantsZippath);

void pass_level0(const torch::jit::Module& mod,
                 std::shared_ptr<torch::jit::Graph>& g,
                 const std::vector<at::Tensor>& input_tensors,
                 const std::vector<at::Tensor>& input_tensors2,
                 const std::vector<std::string>& module_operators,
                 const std::string& ptpath,
                 const std::string& device,
                 std::set<std::string>& foldable_constants,
                 const std::string& foldable_constants_zippath);

}// namespace pnnx

#endif//OPENXAE_PASS_LEVEL0_H
