//
// Created by richard on 6/15/24.
//

#ifndef OPENXAE_LOAD_TORCHSCRIPT_H
#define OPENXAE_LOAD_TORCHSCRIPT_H

#include "Graph.h"

namespace pnnx {

int load_torchscript(
        const std::string& ptpath,
        Graph& pnnx_graph,
        const std::string& device
        );

}

#endif//OPENXAE_LOAD_TORCHSCRIPT_H
