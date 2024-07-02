//
// Created by richard on 7/2/24.
//

#include "reset_device.h"

namespace pnnx {

void ResetDevice(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device) {
    for (auto n: graph->nodes()) {
        if (n->kind().is_aten()) {
            if (n->hasNamedInput("dtype")) {
                auto dtypeNode = n->namedInput("dtype")->node();
                if (dtypeNode->hasAttribute(torch::jit::attr::value)) {
                    // change dtype=half to dtype=float
                    // change dtype=bfloat16 to dtype=float
                    if (dtypeNode->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::i &&
                        (dtypeNode->i(torch::jit::attr::value) == 5 || dtypeNode->i(torch::jit::attr::value) == 15)) {
                        dtypeNode->i_(torch::jit::attr::value, 6);
                    }
                }
            }

            if (n->hasNamedInput("device")) {
                auto deviceNode = n->namedInput("device")->node();
                deviceNode->s_(torch::jit::attr::value, (device == "gpu") ? "cuda" : "cpu");
            }
        }
    }
}

}// namespace pnnx
