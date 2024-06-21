//
// Created by richard on 6/19/24.
//

#include "pass_level0.h"

namespace pnnx {

void pass_level0(const torch::jit::Module& mod,
                 std::shared_ptr<torch::jit::Graph>& g,
                 const std::vector<at::Tensor>& input_tensors,
                 const std::vector<at::Tensor>& input_tensor2,
                 const std::vector<std::string>& module_operators,
                 const std::string& ptpath,
                 const std::string& device,
                 std::set<std::string>& foldable_constants,
                 const std::string& foldable_constants_zippath) {
    //
}

}// namespace pnnx
