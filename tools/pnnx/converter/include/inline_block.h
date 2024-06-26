//
// Created by richard on 6/19/24.
//

#ifndef OPENXAE_INLINE_BLOCK_H
#define OPENXAE_INLINE_BLOCK_H

#include <torch/script.h>

namespace pnnx {

void inline_block(std::shared_ptr<torch::jit::Graph>& graph,
                  const std::vector<std::string>& module_operators);

void InlineBlock(torch::jit::Block* block);
}

#endif//OPENXAE_INLINE_BLOCK_H
