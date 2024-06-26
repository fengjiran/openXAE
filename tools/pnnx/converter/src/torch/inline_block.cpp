//
// Created by richard on 6/19/24.
//

#include "inline_block.h"

#include <deque>
#include <stack>
#include <torch/csrc/api/include/torch/version.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace pnnx {

struct BlockInfo {
    explicit BlockInfo(torch::jit::Block* block) : block_(block) {}
    BlockInfo(torch::jit::Block* block, bool childrenExpanded)
        : block_(block), childrenExpanded_(childrenExpanded) {}

    torch::jit::Block* block_{nullptr};
    bool childrenExpanded_{false};
};

static void inlineCallTo(torch::jit::Node* to_replace, torch::jit::Function* callee) {
    torch::jit::WithInsertPoint guard(to_replace);
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
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
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;) {
        torch::jit::Node* n = *it++;
        if (n->kind() == c10::prim::CallFunction) {
            auto function_constant = n->input(0)->node();
            auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
            if (!fun_type->function()->isGraphFunction()) {
                continue;
            }

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(*(fun_type->function())).graph()->block(), module_operators, inlined_modules, inside_module_op);
#else
            inlineCalls(fun_type->function()->graph()->block(), module_operators, inlined_modules, inside_module_op);
#endif
            n->removeInput(0);
            std::cerr << "inline function " << fun_type->function()->name() << std::endl;
            inlineCallTo(n, fun_type->function());
        } else if (n->kind() == c10::prim::CallMethod) {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            if (!class_type) {
                continue;
            }
            const std::string& function_name = n->s(torch::jit::attr::name);
            torch::jit::Function& function = class_type->getMethod(function_name);
            if (!function.isGraphFunction()) {
                continue;
            }

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());
            std::string class_type_str_no_torch_prefix = class_type_str.substr(10);
            if (!inside_module_op) {
                if (std::find(module_operators.begin(), module_operators.end(), class_type_str_no_torch_prefix) != module_operators.end()) {
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    inlineCalls(toGraphFunction(function).graph()->block(), module_operators, inlined_modules, true);
#else
                    inlineCalls(function.graph()->block(), module_operators, inlined_modules, true);
#endif
                    continue;
                }

                //                                bool skip_inline = false;
                //                                for (const auto& ow: get_global_pnnx_fuse_module_passes()) {
                //                                    if (class_type_str == ow->match_type_str()) {
                //                                        skip_inline = true;
                //                                        break;
                //                                    }
                //                                }
                //
                //                                if (skip_inline)
                //                                    continue;
            }

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(function).graph()->block(), module_operators, inlined_modules, inside_module_op);
#else
            inlineCalls(function.graph()->block(), module_operators, inlined_modules, inside_module_op);
#endif
            inlined_modules.insert(class_type_str_no_torch_prefix);
            inlineCallTo(n, &function);
        } else {
            for (auto b: n->blocks()) {
                inlineCalls(b, module_operators, inlined_modules, inside_module_op);
            }
        }
    }
}

static void ExpandBlock(torch::jit::Block* block, std::stack<BlockInfo>& stk) {
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;) {
        auto n = *it++;
        if (n->kind() == c10::prim::CallFunction) {
            auto function_constant = n->input(0)->node();
            auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
            if (!fun_type->function()->isGraphFunction()) {
                continue;
            }
            stk.emplace(toGraphFunction(*(fun_type->function())).graph()->block());
        } else if (n->kind() == c10::prim::CallMethod) {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            if (!class_type) {
                continue;
            }
            const std::string& function_name = n->s(torch::jit::attr::name);
            torch::jit::Function& function = class_type->getMethod(function_name);
            if (!function.isGraphFunction()) {
                continue;
            }

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());
            std::string class_type_str_no_torch_prefix = class_type_str.substr(10);
            stk.emplace(toGraphFunction(function).graph()->block());
        } else {
            for (auto b: n->blocks()) {
                stk.emplace(b);
            }
        }
    }
}

static void VisitLeafBlock(torch::jit::Block* block) {
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;) {
        auto n = *it++;
        if (n->kind() == c10::prim::CallFunction) {
            auto function_constant = n->input(0)->node();
            auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
            n->removeInput(0);
            std::cerr << "inline function " << fun_type->function()->name() << std::endl;
            inlineCallTo(n, fun_type->function());
        } else if (n->kind() == c10::prim::CallMethod) {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            const std::string& function_name = n->s(torch::jit::attr::name);
            torch::jit::Function& function = class_type->getMethod(function_name);
            inlineCallTo(n, &function);
        }
    }
}

void InlineBlock(torch::jit::Block* block) {
    std::stack<BlockInfo> stk;
    stk.emplace(block);
    while (!stk.empty()) {
        BlockInfo* front = &stk.top();
        if (front->childrenExpanded_) {
            VisitLeafBlock(front->block_);
            stk.pop();
        } else {
            front->childrenExpanded_ = true;
            ExpandBlock(front->block_, stk);
        }
    }
}

void inline_block(std::shared_ptr<torch::jit::Graph>& graph,
                  const std::vector<std::string>& module_operators) {
    std::set<std::string> inlined_modules;

    inlineCalls(graph->block(), module_operators, inlined_modules);

    std::cout << "inlined module num: " << inlined_modules.size() << std::endl;

    for (const auto& x: inlined_modules) {
        if (x == "torch.nn.modules.container.Sequential")
            continue;

        std::cerr << "inline module = " << x.c_str() << std::endl;
    }
}

}// namespace pnnx
