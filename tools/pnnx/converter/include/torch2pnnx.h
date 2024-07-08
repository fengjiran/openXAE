//
// Created by richard on 6/22/24.
//

#ifndef OPENXAE_TORCH2PNNX_H
#define OPENXAE_TORCH2PNNX_H

#include "torch_optimization.h"

namespace pnnx {

int torch2pnnx(const std::string& ptPath,
               Graph& g,
               const std::string& device,
               const std::vector<std::vector<int64_t> >& inputShapes,
               const std::vector<std::string>& inputTypes,
               const std::vector<std::vector<int64_t> >& inputShapes2,
               const std::vector<std::string>& inputTypes2,
               const std::vector<std::string>& customOpModules,
               const std::vector<std::string>& moduleOperators,
               const std::string& foldableConstantsZippath,
               std::set<std::string>& foldableConstants);

}

#endif//OPENXAE_TORCH2PNNX_H
