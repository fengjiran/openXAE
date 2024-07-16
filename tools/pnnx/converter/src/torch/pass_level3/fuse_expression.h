//
// Created by richard on 7/16/24.
//

#ifndef OPENXAE_FUSE_EXPRESSION_H
#define OPENXAE_FUSE_EXPRESSION_H

#include "Graph.h"

namespace pnnx {

void fuse_expression(Graph& graph,
                     const std::set<std::string>& foldableConstants,
                     const std::string& foldableConstantsZippath);

}

#endif//OPENXAE_FUSE_EXPRESSION_H
