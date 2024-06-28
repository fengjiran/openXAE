//
// Created by richard on 6/28/24.
//

#ifndef OPENXAE_CONSTANT_UNPOOLING_H
#define OPENXAE_CONSTANT_UNPOOLING_H

#include <torch/script.h>

namespace pnnx {

void constant_unpooling(std::shared_ptr<torch::jit::Graph>& graph);

}

#endif//OPENXAE_CONSTANT_UNPOOLING_H
