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
            if (std::find(moduleOperators.begin(), moduleOperators.end(), op->type()) == moduleOperators.end()) {
                continue;
            }

            if (op->GetOutputOperands().size() != 1) {
                continue;
            }

            if (op->GetOutputOperands()[0]->GetConsumers().size() != 1) {
                continue;
            }

            const auto& op2 = op->GetOutputOperands()[0]->GetConsumers()[0];
            if (op2->type() != "prim::TupleUnpack") {
                continue;
            }

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
                 const std::shared_ptr<torch::jit::Graph>& tg,
                 const std::vector<std::string>& moduleOperators,
                 Graph& pg) {
    for (int i = 1; i < tg->inputs().size(); ++i) {
        // create pnnx input operator and operand
        std::shared_ptr<Operator> op = pg.CreateOperator("pnnx.Input",
                                                         "pnnx_input_" + std::to_string(i - 1));
        std::shared_ptr<Operand> r = pg.CreateOperand(tg->inputs()[i]);
        r->SetProducer(op);
        op->AddOutputOperand(r);
    }

    std::map<std::string, std::string> classTypeToNames;
    int pnnxUnknownIdx = 0;
    for (const auto n: tg->block()->nodes()) {
        if (n->kind() == c10::prim::GetAttr) {
            std::string name = n->s(torch::jit::attr::name);
            auto classType = n->output(0)->type()->cast<torch::jit::ClassType>();
            if (classType) {
                std::string classTypeStr = classType->str();
                classTypeToNames[classTypeStr] = name;
            } else {
                // Tensor from some class
                std::shared_ptr<Operator> op = pg.CreateOperator("pnnx.Attribute", name);
                for (const auto output: n->outputs()) {
                    std::shared_ptr<Operand> r = pg.CreateOperand(output);
                    r->SetProducer(op);
                    op->AddOutputOperand(r);
                }

                std::deque<std::string> moduleNames;
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name)) {
                        moduleNames.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrapName;
                auto subMod = mod;
                for (const auto& moduleName: moduleNames) {
                    if (!wrapName.empty()) {
                        wrapName += ("." + moduleName);
                    } else {
                        wrapName = moduleName;
                    }
                    subMod = subMod.attr(moduleName).toModule();
                }

                if (wrapName.empty()) {
                    // top-level module
                    wrapName = name;
                }

                op->name() = wrapName;
                op->GetAttributes()["data"] = std::make_shared<Attribute>(subMod.attr(name).toTensor());
                op->GetOutputOperands()[0]->SetType(op->GetAttributes()["data"]->type());
                op->GetOutputOperands()[0]->GetShape() = op->GetAttributes()["data"]->GetShape();
            }
        } else if (n->kind() == c10::prim::Constant) {
            std::string name = "pnnx_" + std::to_string(pnnxUnknownIdx++);
            std::shared_ptr<Operator> op = pg.CreateOperator(n->kind().toDisplayString(), name);
            for (const auto& input: n->inputs()) {
                std::shared_ptr<Operand> r = pg.GetOperand(input->debugName());
                r->AddConsumer(op);
                op->AddInputOperand(r);
            }

            for (const auto& output: n->outputs()) {
                std::shared_ptr<Operand> r = pg.CreateOperand(output);
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
            auto classType = n->input(0)->type()->cast<torch::jit::ClassType>();
            std::string name = classTypeToNames[classType->str()];
            std::string classTypeStr = torch::jit::removeTorchMangle(classType->str());
            std::string classTypeStrNoTorchPrefix = classTypeStr.substr(10);
            std::string opTypeName = classTypeStr;

            for (const auto& ow: FuseModulePassRegistry::GetInstance().GetGlobalPNNXFuseModulePass()) {
                if (classTypeStr == ow->MatchTypeStr()) {
                    opTypeName = ow->TypeStr();
                    break;
                }
            }

            if (opTypeName == classTypeStr) {
                opTypeName = classTypeStrNoTorchPrefix;
            }

            std::shared_ptr<Operator> op = pg.CreateOperator(opTypeName, name);
            for (size_t i = 1; i < n->inputs().size(); i++) {
                std::shared_ptr<Operand> r = pg.GetOperand(n->input(i)->debugName());
                r->AddConsumer(op);
                op->AddInputOperand(r);
            }

            for (const auto output: n->outputs()) {
                std::shared_ptr<Operand> r = pg.CreateOperand(output);
                r->SetProducer(op);
                op->AddOutputOperand(r);
            }

            // module operator
            if (std::find(moduleOperators.begin(),
                          moduleOperators.end(),
                          classTypeStrNoTorchPrefix) != moduleOperators.end()) {
                const std::string& funcName = n->s(torch::jit::attr::name);
                torch::jit::Function& function = classType->getMethod(funcName);
                if (function.isGraphFunction()) {
                    torch::jit::Block* moduleOpBlock = toGraphFunction(function).graph()->block();
                    std::map<size_t, torch::jit::Node*> constAttrNodes;
                    for (const auto mn: moduleOpBlock->nodes()) {
                        if (mn->kind() == c10::prim::GetAttr) {
                            std::string nodeName = mn->s(torch::jit::attr::name);
                            auto classType2 = mn->output(0)->type()->cast<torch::jit::ClassType>();
                            if (!classType2) {
                                std::deque<std::string> moduleNames;// = split(mn->input(0)->node()->s(torch::jit::attr::name), '.');
                                {
                                    auto np = n->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name)) {
                                        moduleNames.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }

                                std::deque<std::string> moduleNames2;
                                {
                                    auto np = mn->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name)) {
                                        moduleNames2.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                for (const auto& x: moduleNames2) {
                                    moduleNames.push_back(x);
                                }

                                auto subMod = mod;
                                for (const auto& module_name: moduleNames) {
                                    subMod = subMod.attr(module_name).toModule();
                                }

                                std::string wrapName;
                                for (const auto& moduleName: moduleNames2) {
                                    if (!wrapName.empty())
                                        wrapName += ("." + moduleName);
                                    else
                                        wrapName = moduleName;
                                }

                                if (wrapName.empty()) {
                                    // top-level module
                                    wrapName = nodeName;
                                } else {
                                    wrapName += ("." + nodeName);
                                }

                                op->GetAttributes()[wrapName] = std::make_shared<Attribute>(subMod.attr(nodeName).toTensor());
                            }
                        } else if (mn->kind() == c10::prim::Constant) {
                            Parameter p = CreateParameterFromTorchNode(mn);
                            if (p.type() == ParameterType::kParameterOther) {
                                size_t uniqueId = mn->output(0)->unique();
                                constAttrNodes[uniqueId] = mn;
                            }
                        }
                    }

                    int pnnxModuleOpUnknownIdx = 0;
                    for (auto attr: constAttrNodes) {
                        char attrName[32];
                        snprintf(attrName, sizeof(attrName), "pnnx_%02d", pnnxModuleOpUnknownIdx);
                        op->GetAttributes()[attrName] = std::make_shared<Attribute>(attr.second->t(torch::jit::attr::value));
                        pnnxModuleOpUnknownIdx++;
                    }
                }
            } else {
                for (const auto& ow: FuseModulePassRegistry::GetInstance().GetGlobalPNNXFuseModulePass()) {
                    if (classTypeStr == ow->MatchTypeStr()) {
                        auto classType2 = n->input(0)->type()->cast<torch::jit::ClassType>();
                        torch::jit::Function& function = classType2->getMethod(n->s(torch::jit::attr::name));

                        std::deque<std::string> moduleNames;
                        {
                            auto np = n->input(0)->node();
                            while (np->hasAttribute(torch::jit::attr::name)) {
                                moduleNames.push_front(np->s(torch::jit::attr::name));
                                np = np->input(0)->node();
                            }
                        }

                        std::string wrapName;
                        auto subMod = mod;
                        for (const auto& moduleName: moduleNames) {
                            if (!wrapName.empty()) {
                                wrapName += ("." + moduleName);
                            } else {
                                wrapName = moduleName;
                            }
                            subMod = subMod.attr(moduleName).toModule();
                        }

                        op->name() = wrapName;
                        ow->Write(op, toGraphFunction(function).graph(), subMod);
                        break;
                    }
                }
            }
        } else {
            std::string name = "pnnx_" + std::to_string(pnnxUnknownIdx++);
            std::shared_ptr<Operator> op = pg.CreateOperator(n->kind().toDisplayString(), name);

            for (const auto input: n->inputs()) {
                std::shared_ptr<Operand> r = pg.GetOperand(input->debugName());
                r->AddConsumer(op);
                op->AddInputOperand(r);
            }

            for (const auto output: n->outputs()) {
                std::shared_ptr<Operand> r = pg.CreateOperand(output);
                r->SetProducer(op);
                op->AddOutputOperand(r);
            }
        }
    }

    for (int i = 0; i < tg->outputs().size(); i++) {
        const auto& in = tg->outputs()[i];
        std::string name = "pnnx_output_" + std::to_string(i);
        std::shared_ptr<Operator> op = pg.CreateOperator("pnnx.Output", name);
        std::shared_ptr<Operand> r = pg.GetOperand(in->debugName());
        r->AddConsumer(op);
        op->AddInputOperand(r);
    }

    // post process
    FuseModuleOpUnpack(pg, moduleOperators);
}

}// namespace pnnx