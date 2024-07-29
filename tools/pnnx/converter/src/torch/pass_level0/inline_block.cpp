//
// Created by richard on 6/19/24.
//
#include "torch/pass_level0.h"
#include "torch/pass_level1.h"
#include <stack>
//#include <torch/csrc/api/include/torch/version.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace pnnx {

static void inlineCallTo(torch::jit::Node* toReplace, torch::jit::Function* callee) {
    torch::jit::WithInsertPoint guard(toReplace);
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> valueMap;

    // TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
    std::vector<torch::jit::Value*> newOutputs = torch::jit::insertGraph(*toReplace->owningGraph(),
                                                                          *(toGraphFunction(*callee).graph()),
                                                                          toReplace->inputs(),
                                                                          valueMap);

    const auto& oldOutputs = toReplace->outputs();
    for (size_t i = 0; i < oldOutputs.size(); ++i) {
        newOutputs[i]->copyMetadata(oldOutputs[i]);
        oldOutputs[i]->replaceAllUsesWith(newOutputs[i]);
    }
    toReplace->destroy();
}

static void inlineCalls(torch::jit::Block* block,
                        const std::vector<std::string>& moduleOperators,
                        std::set<std::string>& inlinedModules,
                        bool insideModuleOp = false) {
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;) {
        torch::jit::Node* n = *it++;
        if (n->kind() == c10::prim::CallFunction) {
            auto functionConstant = n->input(0)->node();
            auto functionType = functionConstant->output()->type()->expect<torch::jit::FunctionType>();
            if (!functionType->function()->isGraphFunction()) {
                continue;
            }

            // TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(*(functionType->function())).graph()->block(),
                        moduleOperators, inlinedModules, insideModuleOp);
            n->removeInput(0);
            std::cerr << "inline function " << functionType->function()->name() << std::endl;
            inlineCallTo(n, functionType->function());
        } else if (n->kind() == c10::prim::CallMethod) {
            auto classType = n->input(0)->type()->cast<torch::jit::ClassType>();
            if (!classType) {
                continue;
            }
            const std::string& functionName = n->s(torch::jit::attr::name);
            torch::jit::Function& function = classType->getMethod(functionName);
            if (!function.isGraphFunction()) {
                continue;
            }

            std::string classTypeStr = torch::jit::removeTorchMangle(classType->str());
            std::string classTypeStrNoTorchPrefix = classTypeStr.substr(10);
            if (!insideModuleOp) {
                if (std::find(moduleOperators.begin(), moduleOperators.end(), classTypeStrNoTorchPrefix) !=
                    moduleOperators.end()) {

                    // TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    inlineCalls(toGraphFunction(function).graph()->block(),
                                moduleOperators, inlinedModules, true);
                    continue;
                }

                bool skipInline = false;
                std::cerr << "module pass num: " << FuseModulePassRegistry::GetInstance().GetGlobalPNNXFuseModulePass().size() << "\n";
                for (const auto& ow: FuseModulePassRegistry::GetInstance().GetGlobalPNNXFuseModulePass()) {
                    if (classTypeStr == ow->MatchTypeStr()) {
                        skipInline = true;
                        break;
                    }
                }

                if (skipInline) {
                    continue;
                }
            }

            // TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(function).graph()->block(),
                        moduleOperators, inlinedModules, insideModuleOp);
            inlinedModules.insert(classTypeStrNoTorchPrefix);
            inlineCallTo(n, &function);
        } else {
            for (auto b: n->blocks()) {
                inlineCalls(b, moduleOperators, inlinedModules, insideModuleOp);
            }
        }
    }
}

void inline_block(std::shared_ptr<torch::jit::Graph>& graph,
                  const std::vector<std::string>& moduleOperators) {
    std::set<std::string> inlinedModules;

    inlineCalls(graph->block(), moduleOperators, inlinedModules);

    std::cout << "inlined module num: " << inlinedModules.size() << std::endl;

    for (const auto& x: inlinedModules) {
        if (x == "torch.nn.modules.container.Sequential")
            continue;

        std::cerr << "inline module = " << x << std::endl;
    }
}

class BlockInfo {
public:
    explicit BlockInfo(torch::jit::Block* block) : block_(block) {}

    torch::jit::Block* data() {
        return block_;
    }

    bool AllChildrenInlined() const {
        return allChildrenInlined_;
    }

    void SetChildrenInlinedStatus(bool allChildrenInlined) {
        allChildrenInlined_ = allChildrenInlined;
    }

    void AddCallableNode(const std::pair<torch::jit::Node*, std::shared_ptr<c10::NamedType>>& nf) {
        callableNodes_.push_back(nf);
    }

    const auto& GetCallableNodes() const {
        return callableNodes_;
    }

private:
    torch::jit::Block* block_{nullptr};
    bool allChildrenInlined_{false};
    std::vector<std::pair<torch::jit::Node*, std::shared_ptr<c10::NamedType>>> callableNodes_;
};

static void ExpandBlock(BlockInfo& block, std::stack<BlockInfo>& stk) {
    for (auto n: block.data()->nodes()) {
        if (n->kind() == c10::prim::CallFunction) {
            auto function_constant = n->input(0)->node();
            auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
            if (!fun_type->function()->isGraphFunction()) {
                continue;
            }

            block.AddCallableNode({n, fun_type});
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

            block.AddCallableNode({n, class_type});
            stk.emplace(toGraphFunction(function).graph()->block());
        } else {
            for (auto b: n->blocks()) {
                stk.emplace(b);
            }
        }
    }
}

static void InlineLeafBlock(const BlockInfo& block, std::set<std::string>& inlinedModules) {
    for (auto [node, obj]: block.GetCallableNodes()) {
        if (node->kind() == c10::prim::CallFunction) {
            auto func = std::dynamic_pointer_cast<c10::FunctionType>(obj);
            node->removeInput(0);
            std::cerr << "inline function: " << func->function()->name() << std::endl;
            inlineCallTo(node, func->function());
        } else if (node->kind() == c10::prim::CallMethod) {
            auto cls = std::dynamic_pointer_cast<c10::ClassType>(obj);
            const std::string& name = node->s(torch::jit::attr::name);
            torch::jit::Function& method = cls->getMethod(name);

            std::string classTypeStr = torch::jit::removeTorchMangle(cls->str());
            std::string classTypeWithNoTorchPrefix = classTypeStr.substr(10);
            inlinedModules.insert(classTypeWithNoTorchPrefix);

            std::cerr << "inline CallMethod: " << classTypeWithNoTorchPrefix << std::endl;
            inlineCallTo(node, &method);
        }
    }
}

void InlineBlock(torch::jit::Block* block,
                 const std::vector<std::string>& moduleOps,
                 std::set<std::string>& inlinedModules,
                 bool insideModuleOp = false) {
    std::stack<BlockInfo> stk;
    stk.emplace(block);
    while (!stk.empty()) {
        BlockInfo& top = stk.top();
        if (top.AllChildrenInlined()) {
            InlineLeafBlock(top, inlinedModules);
            stk.pop();
        } else {
            top.SetChildrenInlinedStatus(true);
            ExpandBlock(top, stk);
        }
    }
}

void Inline(std::shared_ptr<torch::jit::Graph>& graph,
            const std::vector<std::string>& moduleOps) {
    std::set<std::string> inlinedModules;

    InlineBlock(graph->block(), moduleOps, inlinedModules);

    std::cout << "inlined module num: " << inlinedModules.size() << std::endl;

    for (const auto& x: inlinedModules) {
        if (x == "torch.nn.modules.container.Sequential")
            continue;

        std::cerr << "inline module = " << x << std::endl;
    }
}

}// namespace pnnx
