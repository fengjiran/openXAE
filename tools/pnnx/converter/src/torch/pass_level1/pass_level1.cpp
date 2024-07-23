//
// Created by richard on 7/17/24.
//

#include "torch/pass_level1.h"
#include "torch/torch2pnnx.h"

#include <torch/csrc/jit/passes/quantization/helper.h>

namespace pnnx {

static void FuseModuleOpUnpack(Graph& graph, const std::vector<std::string>& moduleOperators) {
    while (true) {
        bool matched = false;
        for (auto& op: graph.GetOperators()) {
            if (std::find(moduleOperators.begin(), moduleOperators.end(), op->type()) == moduleOperators.end())
                continue;

            if (op->GetOutputOperands().size() != 1)
                continue;

            if (op->GetOutputOperands()[0]->GetConsumers().size() != 1)
                continue;

            const auto& op2 = op->GetOutputOperands()[0]->GetConsumers()[0];
            if (op2->type() != "prim::TupleUnpack")
                continue;

            matched = true;

            op->GetOutputOperands()[0]->SetProducer({});
            op->GetOutputOperands()[0]->RemoveConsumer(op2);

            for (auto& x: op2->GetOutputOperands()) {
                x->SetProducer(op);
            }

            op->GetOutputOperands() = op2->GetOutputOperands();

            op2->GetInputOperands().clear();
            op2->GetOutputOperands().clear();

            graph.GetOperators().erase(
                    std::find(graph.GetOperators().begin(), graph.GetOperators().end(), op2));

            break;
        }

        if (!matched)
            break;
    }
}

void pass_level1(const torch::jit::Module& mod,
                 const std::shared_ptr<torch::jit::Graph>& g,
                 const std::vector<std::string>& module_operators,
                 Graph& pg) {
    for (int i = 1; i < (int) g->inputs().size(); i++) {
        const auto in = g->inputs()[i];

        char name[32];
        snprintf(name, sizeof(name), "pnnx_input_%d", i - 1);

        // create pnnx operator and operand
        std::shared_ptr<Operator> op = pg.CreateOperator("pnnx.Input", name);
        std::shared_ptr<Operand> r = pg.CreateOperand(in);
        r->SetProducer(op);
        op->AddOutputOperand(r);
    }

    std::map<std::string, std::string> class_type_to_names;
    int pnnx_unknown_index = 0;

    for (const auto& n: g->block()->nodes()) {
        if (n->kind() == c10::prim::GetAttr) {
            // pass
            std::string name = n->s(torch::jit::attr::name);
            //             std::string name = n->debugName();

            auto class_type = n->output(0)->type()->cast<torch::jit::ClassType>();

            if (class_type) {
                std::string class_type_str = class_type->str();
                class_type_to_names[class_type_str] = name;
            } else {
                // Tensor from some class
                std::shared_ptr<Operator> op = pg.CreateOperator("pnnx.Attribute", name);

                for (int i = 0; i < (int) n->outputs().size(); i++) {
                    const auto on = n->output(i);
                    std::shared_ptr<Operand> r = pg.CreateOperand(on);
                    r->SetProducer(op);
                    op->AddOutputOperand(r);
                }

                std::deque<std::string> moduleNames;// = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name)) {
                        moduleNames.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrappedName;
                auto subMod = mod;
                for (const auto& moduleName: moduleNames) {
                    if (!wrappedName.empty())
                        wrappedName += ("." + moduleName);
                    else
                        wrappedName = moduleName;
                    subMod = subMod.attr(moduleName).toModule();
                }

                if (wrappedName.empty()) {
                    // top-level module
                    wrappedName = name;
                }

                op->name() = wrappedName;

                // op->params["this"] = n->input(i)

                // sub_mod.dump(true, true, true);

                op->GetAttributes()["data"] = std::make_shared<Attribute>(subMod.attr(name).toTensor());
                op->GetOutputOperands()[0]->SetType(op->GetAttributes()["data"]->type());
                op->GetOutputOperands()[0]->GetShape() = op->GetAttributes()["data"]->GetShape();
            }
        } else if (n->kind() == c10::prim::Constant) {// || n->kind() == c10::prim::ListConstruct)
            char name[32];
            snprintf(name, sizeof(name), "pnnx_%d", pnnx_unknown_index++);

            std::shared_ptr<Operator> op = pg.CreateOperator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int) n->inputs().size(); i++) {
                const auto& in = n->input(i);
                std::shared_ptr<Operand> r = pg.GetOperand(in->debugName());
                r->AddConsumer(op);
                op->AddInputOperand(r);
            }

            for (int i = 0; i < (int) n->outputs().size(); i++) {
                const auto& on = n->output(i);
                std::shared_ptr<Operand> r = pg.CreateOperand(on);
                r->SetProducer(op);
                op->AddOutputOperand(r);
            }

            op->GetParameters()["value"] = std::make_shared<Parameter>(CreateParameterFromTorchNode(n));

            if (op->GetParameters()["value"]->type() == ParameterType::kParameterOther) {
                op->type() = "pnnx.Attribute";
                op->GetParameters().erase("value");
                op->GetAttributes()["data"] = std::make_shared<Attribute>(n->t(torch::jit::attr::value));
            }
        } else if (n->kind() == c10::prim::CallMethod) {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            //             const std::string& name = n->s(torch::jit::attr::name);

            //             fprintf(stderr, "call %s\n", class_type->str().c_str());

            std::string name = class_type_to_names[class_type->str()];

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());

            std::string class_type_str_no_torch_prefix = class_type_str.substr(10);

            std::string optypename = class_type_str;

            for (const auto& ow: FuseModulePassRegistry::GetInstance().GetGlobalPNNXFuseModulePass()) {
                if (class_type_str != ow->MatchTypeStr())
                    continue;

                optypename = ow->TypeStr();
                break;
            }

            if (optypename == class_type_str) {
                optypename = class_type_str_no_torch_prefix;
            }

            std::shared_ptr<Operator> op = pg.CreateOperator(optypename, name);

            for (int i = 1; i < (int) n->inputs().size(); i++) {
                const auto& in = n->input(i);
                std::shared_ptr<Operand> r = pg.GetOperand(in->debugName());
                r->AddConsumer(op);
                op->AddInputOperand(r);
            }

            for (int i = 0; i < (int) n->outputs().size(); i++) {
                const auto& on = n->output(i);
                std::shared_ptr<Operand> r = pg.CreateOperand(on);
                r->SetProducer(op);
                op->AddOutputOperand(r);
            }

            // module operator
            if (std::find(module_operators.begin(), module_operators.end(), class_type_str_no_torch_prefix) != module_operators.end()) {
                const std::string& function_name = n->s(torch::jit::attr::name);
                torch::jit::Function& function = class_type->getMethod(function_name);
                if (function.isGraphFunction()) {
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    torch::jit::Block* moduleop_block = toGraphFunction(function).graph()->block();
#else
                    torch::jit::Block* moduleop_block = function.graph()->block();
#endif

                    std::map<size_t, torch::jit::Node*> constant_attr_nodes;
                    for (const auto& mn: moduleop_block->nodes()) {
                        if (mn->kind() == c10::prim::GetAttr) {
                            std::string name = mn->s(torch::jit::attr::name);
                            //             std::string name = mn->debugName();

                            auto class_type = mn->output(0)->type()->cast<torch::jit::ClassType>();

                            if (!class_type) {
                                std::deque<std::string> module_names;// = split(mn->input(0)->node()->s(torch::jit::attr::name), '.');
                                {
                                    auto np = n->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name)) {
                                        module_names.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                std::deque<std::string> module_names2;
                                {
                                    auto np = mn->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name)) {
                                        module_names2.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                for (const auto& x: module_names2) {
                                    module_names.push_back(x);
                                }

                                auto sub_mod = mod;
                                for (const auto& module_name: module_names) {
                                    sub_mod = sub_mod.attr(module_name).toModule();
                                }

                                std::string wrapped_name;
                                for (const auto& module_name: module_names2) {
                                    if (!wrapped_name.empty())
                                        wrapped_name += ("." + module_name);
                                    else
                                        wrapped_name = module_name;
                                }

                                if (wrapped_name.empty()) {
                                    // top-level module
                                    wrapped_name = name;
                                } else {
                                    wrapped_name += ("." + name);
                                }

                                op->GetAttributes()[wrapped_name] = std::make_shared<Attribute>(sub_mod.attr(name).toTensor());
                            }
                        } else if (mn->kind() == c10::prim::Constant) {
                            Parameter p(mn);

                            if (p.type() == ParameterType::kParameterOther) {
                                size_t unique_id = mn->output(0)->unique();
                                constant_attr_nodes[unique_id] = mn;
                            }
                        }
                    }

                    int pnnx_moduleop_unknown_index = 0;
                    for (auto attr: constant_attr_nodes) {
                        char name[32];
                        sprintf(name, "pnnx_%02d", pnnx_moduleop_unknown_index);
                        op->GetAttributes()[name] = std::make_shared<Attribute>(attr.second->t(torch::jit::attr::value));
                        pnnx_moduleop_unknown_index++;
                    }
                }
            } else {
                for (const auto& ow: FuseModulePassRegistry::GetInstance().GetGlobalPNNXFuseModulePass()) {
                    if (class_type_str != ow->MatchTypeStr())
                        continue;

                    auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
                    torch::jit::Function& function = class_type->getMethod(n->s(torch::jit::attr::name));

                    std::deque<std::string> module_names;// = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                    {
                        auto np = n->input(0)->node();
                        while (np->hasAttribute(torch::jit::attr::name)) {
                            module_names.push_front(np->s(torch::jit::attr::name));
                            np = np->input(0)->node();
                        }
                    }

                    std::string wrapped_name;
                    auto sub_mod = mod;
                    for (const auto& module_name: module_names) {
                        if (!wrapped_name.empty())
                            wrapped_name += ("." + module_name);
                        else
                            wrapped_name = module_name;
                        sub_mod = sub_mod.attr(module_name).toModule();
                    }

                    op->name() = wrapped_name;

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    ow->Write(op, toGraphFunction(function).graph(), sub_mod);
#else
                    ow->Write(op, function.graph(), sub_mod);
#endif

                    break;
                }
            }
        }
        // else if (n->kind() == c10::prim::CallFunction)
        // {
        //     fprintf(stderr, "function %s", n->kind().toDisplayString());
        //
        //     AT_ASSERT(cur->input(0)->node()->kind() == c10::prim::Constant);
        //     auto function_constant = cur->input(0)->node();
        //     auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
        //     if (!fun_type->function()->isGraphFunction())
        //     {
        //         continue;
        //     }
        //     cur->removeInput(0);
        //
        //     fprintf(stderr, "inline function %s\n", fun_type->function()->name().c_str());
        //
        //     GRAPH_UPDATE("Inlining function '", fun_type->function()->name(), "' to ", *cur);
        //     GRAPH_UPDATE("Function body: ", *fun_type->function()->optimized_graph());
        //     inlineCallTo(cur, fun_type->function(), false);
        //     break;
        // }
        else {
            char name[32];
            snprintf(name, sizeof(name), "pnnx_%d", pnnx_unknown_index++);

            std::shared_ptr<Operator> op = pg.CreateOperator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int) n->inputs().size(); i++) {
                const auto& in = n->input(i);
                std::shared_ptr<Operand> r = pg.GetOperand(in->debugName());
                r->AddConsumer(op);
                op->AddInputOperand(r);
            }

            for (int i = 0; i < (int) n->outputs().size(); i++) {
                const auto& on = n->output(i);
                std::shared_ptr<Operand> r = pg.CreateOperand(on);
                r->SetProducer(op);
                op->AddOutputOperand(r);
            }
        }
    }

    for (int i = 0; i < (int) g->outputs().size(); i++) {
        const auto& in = g->outputs()[i];

        char name[32];
        snprintf(name, sizeof(name), "pnnx_output_%d", i);
        std::shared_ptr<Operator> op = pg.CreateOperator("pnnx.Output", name);
        std::shared_ptr<Operand> r = pg.GetOperand(in->debugName());
        r->AddConsumer(op);
        op->AddInputOperand(r);
    }

    // post process
    FuseModuleOpUnpack(pg, module_operators);
}

}// namespace pnnx