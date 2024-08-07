//
// Created by richard on 8/6/24.
//

#include "pnnx/converter/include/torch/pass_level2.h"

namespace pnnx {

void GraphRewriterPass::Write(const std::shared_ptr<Operator>& op,
                              const std::map<std::string, Parameter>& capturedParams) const {
    //
}

static bool IsAliasOp(const std::shared_ptr<Operator>& op) {
    if (op->type() == "aten::slice" ||
        op->type() == "aten::select" ||
        op->type() == "aten::view") {
        return true;
    }

    return false;
}

}// namespace pnnx
