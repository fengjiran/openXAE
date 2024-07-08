//
// Created by richard on 7/8/24.
//

#ifndef OPENXAE_CONVERT_HALF_TO_FLOAT_H
#define OPENXAE_CONVERT_HALF_TO_FLOAT_H

#include <torch/script.h>

namespace pnnx {

void ConvertHalf2Float(torch::jit::Module& mod);

}

#endif//OPENXAE_CONVERT_HALF_TO_FLOAT_H
