//
// Created by richard on 7/2/24.
//

#include "torch/pass_level0.h"

namespace pnnx {

void FlattenInput(std::shared_ptr<torch::jit::Graph>& graph) {
    while (true) {
        bool matched = false;
        for (auto n: graph->nodes()) {
            if (n->kind() != c10::prim::TupleUnpack && n->kind() != c10::prim::ListUnpack) {
                continue;
            }

            for (size_t i = 0; i < graph->inputs().size(); ++i) {
                if (n->input(0) == graph->inputs()[i]) {
                    matched = true;
                    for (size_t j = 0; j < n->outputs().size(); ++j) {
                        auto v2 = graph->insertInput(i + 1 + j);
                        n->output(j)->replaceAllUsesWith(v2);
                    }
                    n->destroy();
                    graph->eraseInput(i);
                    break;
                }
            }
            if (matched) {
                break;
            }
        }
        if (!matched) {
            break;
        }
    }
}

}// namespace pnnx
