//
// Created by richard on 6/22/24.
//

#ifndef OPENXAE_TORCH2PNNX_H
#define OPENXAE_TORCH2PNNX_H

#include "torch_optimization.h"

namespace pnnx {

int torch2pnnx(const std::string& ptPath,
               const std::string& device);

}

#endif//OPENXAE_TORCH2PNNX_H
