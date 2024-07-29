//
// Created by richard on 7/19/24.
//
#include "torch/pass_level0.h"
#include <torch/csrc/jit/passes/inliner.h>

namespace pnnx {

void pass_level0(const torch::jit::Module& mod,
                 std::shared_ptr<torch::jit::Graph>& g,
                 const std::vector<at::Tensor>& inputTensors,
                 const std::vector<at::Tensor>& inputTensors2,
                 const std::vector<std::string>& moduleOperators,
                 const std::string& ptPath,
                 const std::string& device,
                 std::set<std::string>& foldableConstants,
                 const std::string& foldableConstantsZippath) {
    //    inline_block(g, moduleOperators);
    Inline(*g);
    ConstantUnpooling(g);
    ResetDevice(g, device);
    FlattenInput(g);
    if (!inputTensors.empty()) {
        ShapeInference(mod, g, inputTensors, inputTensors2, moduleOperators, ptPath,
                       device, foldableConstants, foldableConstantsZippath);
    }
}

}// namespace pnnx
