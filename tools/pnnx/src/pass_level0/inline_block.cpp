//
// Created by richard on 6/19/24.
//

#include "inline_block.h"

namespace pnnx {

static void inlineCallTo(torch::jit::Node* to_replace, torch::jit::Function* callee) {
    torch::jit::WithInsertPoint guard(to_replace);
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;

#if Torch_VERSION_MAJOR >= 2 || (Torch_VERSION_MAJOR >= 1 && Torch_VERSION_MINOR >= 11)
    std::vector<torch::jit::Value*> new_outputs = torch::jit::insertGraph(*to_replace->owningGraph(),
                                                                          *(toGraphFunction(*callee).graph()),
                                                                          to_replace->inputs(),
                                                                          value_map);
#else
    std::vector<torch::jit::Value*> new_outputs = torch::jit::insertGraph(*to_replace->owningGraph(),
                                                                          *(callee->graph()),
                                                                          to_replace->inputs(),
                                                                          value_map);
#endif

    const auto& old_outputs = to_replace->outputs();
    for (size_t i = 0; i < old_outputs.size(); ++i) {
        new_outputs[i]->copyMetadata(old_outputs[i]);
        old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
    }
    to_replace->destroy();
}

static void inlineCalls(torch::jit::Block* block,
                        const std::vector<std::string>& module_operators,
                        std::set<std::string>& inlined_modules,
                        bool inside_module_op = false) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
        torch::jit::Node* n = *it;
        if (n->kind() == c10::prim::CallFunction) {
            auto function_constant = n->input(0)->node();
            auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
            if (!fun_type->function()->isGraphFunction()) {
                continue;
            }

#if Torch_VERSION_MAJOR >= 2 || (Torch_VERSION_MAJOR >= 1 && Torch_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(*(fun_type->function())).graph()->block(), module_operators, inlined_modules, inside_module_op);
#else
            inlineCalls(fun_type->function()->graph()->block(), module_operators, inlined_modules, inside_module_op);
#endif
            n->removeInput(0);
            std::cerr << "inline function " << fun_type->function()->name() << std::endl;
            inlineCallTo(n, fun_type->function());
        } else if (n->kind() == c10::prim::CallMethod) {
            //
        }
    }
}

void inline_block(std::shared_ptr<torch::jit::Graph>& graph,
                  const std::vector<std::string>& module_operators) {
    //
}

}// namespace pnnx
