//
// Created by richard on 7/2/24.
//

#ifndef OPENXAE_FLATTEN_INPUT_H
#define OPENXAE_FLATTEN_INPUT_H

#include <torch/script.h>

namespace pnnx {

void FlattenInput(std::shared_ptr<torch::jit::Graph>& graph);

}

#endif//OPENXAE_FLATTEN_INPUT_H
