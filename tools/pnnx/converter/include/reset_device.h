//
// Created by richard on 7/2/24.
//

#ifndef OPENXAE_RESET_DEVICE_H
#define OPENXAE_RESET_DEVICE_H

#include <torch/script.h>

namespace pnnx {

void ResetDevice(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device);

}

#endif//OPENXAE_RESET_DEVICE_H
