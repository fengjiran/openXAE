//
// Created by richard on 8/6/24.
//

#include "pnnx/converter/include/torch/pass_level2.h"

namespace pnnx {

void GraphRewriterPass::Write(const std::shared_ptr<Operator>& op,
                              const std::map<std::string, Parameter>& capturedParams) const {
    if (ReplacePatternGraph().empty()) {
        for (const auto& x: capturedParams) {
            op->GetParameters()[x.first] = std::make_shared<Parameter>(x.second);
        }
        return;
    }

    for (const auto& x: op->GetParameters()) {
        if (x.second->type() != ParameterType::kParameterString) {
            continue;
        }

        std::string str = x.second->toString();
        if (str.find('%') == std::string::npos) {
            continue;
        }

        // search % token and replace with captured
        size_t pos = str.find('%');
        while (pos != std::string::npos) {
            // %xyz
            char buf[256];
            sscanf(str.c_str() + pos + 1, "%255[^][,() ]", buf);
            std::string key(buf);
            if (capturedParams.find(key) == capturedParams.end()) {
                std::cerr << "replace pattern param " << key << " missing captured.\n";
                return;
            }

            // replace %xyz with encoded_str
            auto encodedStr = capturedParams.at(key).toString();
            str.replace(pos, key.size() + 1, encodedStr);
            pos = str.find('%', pos + 1);
        }

        op->GetParameters()[x.first] = std::make_shared<Parameter>(
                Parameter::CreateParameterFromString(str));
    }

    for (const auto& operand: op->GetInputOperands()) {
        auto shape = operand->GetShape();
        for (size_t i = 0; i < shape.size(); ++i) {
            int ai = shape[i];
            if (ai == DimVariableTag) {
                auto key = operand->GetParams().at(std::string("__shape__") + std::to_string(i))->toValue<std::string>();
                if (capturedParams.find(key) == capturedParams.end()) {
                    std::cerr << "replace pattern param " << key << " missing captured.\n";
                    return;
                }

                shape[i] = capturedParams.at(key).toValue<int>();
            }
        }
    }

    for (const auto& operand: op->GetOutputOperands()) {
        auto shape = operand->GetShape();
        for (size_t i = 0; i < shape.size(); ++i) {
            int ai = shape[i];
            if (ai == DimVariableTag) {
                auto key = operand->GetParams().at(std::string("__shape__") + std::to_string(i))->toValue<std::string>();
                if (capturedParams.find(key) == capturedParams.end()) {
                    std::cerr << "replace pattern param " << key << " missing captured.\n";
                    return;
                }

                shape[i] = capturedParams.at(key).toValue<int>();
            }
        }
    }
}

void GraphRewriterPass::Write(const std::shared_ptr<Operator>& op,
                              const std::map<std::string, Parameter>& capturedParams,
                              const std::map<std::string, Attribute>& capturedAttrs) const {
    Write(op, capturedParams);
    for (auto& x: op->GetAttributes()) {
        if (x.second->type() != DataType::kDataTypeUnknown) {
            continue;
        }

        std::string key(x.second->GetRawData().data());
        if (key.empty()) {
            continue;
        }

        //        op->GetAttributes()[x.first] = std::make_shared<Attribute>(capturedAttrs.at(key));
        x.second = std::make_shared<Attribute>(capturedAttrs.at(key));
    }
}

void GraphRewriterPass::Write(const std::map<std::string, std::shared_ptr<Operator>>& ops,
                              const std::map<std::string, Parameter>& capturedParams) const {
    for (const auto& x: ops) {
        Write(x.second, capturedParams);
    }
}

void GraphRewriterPass::Write(const std::map<std::string, std::shared_ptr<Operator>>& ops,
                              const std::map<std::string, Parameter>& capturedParams,
                              const std::map<std::string, Attribute>& capturedAttrs) const {
    Write(ops, capturedParams);
    for (const auto& x: ops) {
        for (auto& attr: x.second->GetAttributes()) {
            if (attr.second->type() != DataType::kDataTypeUnknown) {
                continue;
            }

            std::string key(attr.second->GetRawData().data());
            if (key.empty() || key[0] != '%') {
                continue;
            }

            attr.second = std::make_shared<Attribute>(capturedAttrs.at(key.substr(1)));
        }
    }
}

static bool IsAliasOp(const std::shared_ptr<Operator>& op) {
    if (op->type() == "aten::slice" ||
        op->type() == "aten::select" ||
        op->type() == "aten::view") {
        return true;
    }

    return false;
}

static void functionize(Graph& graph) {

    // step1: create shadow view/slice/select for each consumer.
    {
        for (int i = (int) graph.GetOperators().size() - 1; i >= 0; --i) {
            auto op = graph.GetOperators()[i];
            if (!IsAliasOp(op)) {
                continue;
            }

            auto out0 = op->GetOutputOperands()[0];
            if (out0->GetConsumers().size() == 1) {
                continue;
            }

            for (int j = (int) out0->GetConsumers().size() - 1; j > 0; --j) {
                auto op1 = out0->GetConsumers()[j];
                auto shadowOp = graph.CreateOperatorAfter(op->type(),
                                                          op->name() + "_pnnxshadow_" + std::to_string(j),
                                                          op);
                auto shadowOut = graph.CreateOperand(shadowOp->name() + "_out");

                shadowOp->GetInputOperands() = op->GetInputOperands();
                shadowOp->GetParameters() = op->GetParameters();
                shadowOp->AddOutputOperand(shadowOut);

                for (const auto& x: op->GetInputOperands()) {
                    x->AddConsumer(shadowOp);
                }

                shadowOut->SetProducer(shadowOp);
                shadowOut->SetType(out0->type());
                shadowOut->GetShape() = out0->GetShape();
                shadowOut->GetParams() = out0->GetParams();
                shadowOut->AddConsumer(op1);

                for (auto& operand: op1->GetInputOperands()) {
                    if (operand == out0) {
                        operand = shadowOut;
                    }
                }
            }

            out0->GetConsumers().resize(1);
        }
    }

    // step2: replace inplace op, append copy
    // step3: tag operand alias for view/slice/select/... output
    {
        for (const auto& op: graph.GetOperators()) {
            bool isInplaceOp = op->type().size() > 2 &&
                               op->type()[op->type().size() - 2] != '_' &&
                               op->type()[op->type().size() - 1] == '_';
            if (op->type() != "aten::copy_" && !IsAliasOp(op) && !isInplaceOp) {
                continue;
            }

            auto in = op->GetInputOperands()[0];
            int aliasIdx;
            if (in->GetParams().find("__alias__") != in->GetParams().end()) {
                aliasIdx = in->GetParams().at("__alias__")->toValue<int>();
            } else {
                aliasIdx = std::find(graph.GetOperands().begin(), graph.GetOperands().end(), in) -
                           graph.GetOperands().begin();
            }

            if (op->type() == "aten::copy_") {
                op->GetOutputOperands()[0]->GetParams()["__alias__"] = std::make_shared<Parameter>(aliasIdx);

                // set copy output shape as the alias one
                op->GetOutputOperands()[0]->SetType(graph.GetOperands()[aliasIdx]->type());
                op->GetOutputOperands()[0]->GetShape() = graph.GetOperands()[aliasIdx]->GetShape();
                continue;
            }

            if (IsAliasOp(op)) {
                op->GetOutputOperands()[0]->GetParams()["__alias__"] = std::make_shared<Parameter>(aliasIdx);
                continue;
            }

            if (isInplaceOp) {
                // replace with non-inplace version, create copy op
                op->type() = op->type().substr(0, op->type().size() - 1);

                // append aten::copy_
                if (graph.GetOperands()[aliasIdx]->GetConsumers().size() > 1) {
                    auto in0 = op->GetInputOperands()[0];
                    auto out0 = op->GetOutputOperands()[0];

                    auto copyOp = graph.CreateOperatorAfter("aten::copy_", op->name() + "_copy", op);
                    auto copyOut = graph.CreateOperand(op->name() + "_copy_out");

                    copyOp->AddInputOperand(in0);
                    copyOp->AddInputOperand(out0);
                    in0->AddConsumer(copyOp);
                    out0->AddConsumer(copyOp);

                    copyOp->AddOutputOperand(copyOut);
                    copyOut->SetProducer(copyOp);
                }
            }
        }
    }

    // step4: scan inplace copy op, collect affected alias
    {
        for (size_t i = 0; i < graph.GetOperators().size(); ++i) {
            auto op = graph.GetOperators()[i];
            if (op->type() != "aten::copy_") {
                continue;
            }

            op->type() = "aten::copy";
            auto out0 = op->GetOutputOperands()[0];

            // inplace op output always alias with the input
            const int aliasIdx = out0->GetParams().at("__alias__")->toValue<int>();
            auto aliasIn0 = graph.GetOperands()[aliasIdx];

            size_t iAdvanced = 0;

            // step5: look fpr any op after the inplace op with alias input
            for (size_t j = i + 1; j < graph.GetOperators().size(); ++j) {
                auto op1 = graph.GetOperators()[j];
                bool affected = false;
                for (const auto& x: op1->GetInputOperands()) {
                    if (x == aliasIn0) {
                        affected = true;
                        break;
                    }

                    if (x->GetParams().find("__alias__") == x->GetParams().end()) {
                        continue;
                    }

                    int aliasIdx1 = x->GetParams().at("__alias__")->toValue<int>();
                    if (aliasIdx1 == aliasIdx) {
                        affected = true;
                        break;
                    }
                }

                if (!affected) {
                    continue;
                }

                // step6: collect ops on the chain back to alias
                std::set<size_t> chainsx_op_indexes;
                {
                    size_t op1Idx = std::find(graph.GetOperators().begin(), graph.GetOperators().end(), op1) -
                                    graph.GetOperators().begin();
                    if (op1Idx < i - iAdvanced) {
                        chainsx_op_indexes.insert(op1Idx);
                    }

                    while (true) {
                        auto x = op1->GetInputOperands()[0];
                        if (x->GetParams().find("__alias__") == x->GetParams().end()) {
                            break;
                        }

                        int aliasIdx1 = x->GetParams().at("__alias__")->toValue<int>();
                        if (aliasIdx1 != aliasIdx) {
                            break;
                        }

                        op1 = x->GetProducer();
                        size_t newOp1Idx = std::find(graph.GetOperators().begin(), graph.GetOperators().end(), op1) -
                                           graph.GetOperators().begin();
                        if (newOp1Idx < i - iAdvanced) {
                            chainsx_op_indexes.insert(newOp1Idx);
                        }
                    }
                }

                // step7: move chain after copy op
                {
                    int k = 0;
                    for (size_t doi: chainsx_op_indexes) {
                        doi -= k;

                        for (size_t l = doi; l < i - iAdvanced; ++l) {
                            std::swap(graph.GetOperators()[l], graph.GetOperators()[l + 1]);
                        }

                        k++;
                    }
                    iAdvanced += chainsx_op_indexes.size();
                }

                // step8: update all alias uses after copy op, retag alias
                out0->GetParams().erase("__alias__");
                const int newAliasIdx = std::find(graph.GetOperands().begin(), graph.GetOperands().end(), out0) -
                                        graph.GetOperands().begin();
                for (size_t k = i - iAdvanced + 1; k < graph.GetOperators().size(); ++k) {
                    auto op2 = graph.GetOperators()[k];

                    for (size_t l = 0; l < op2->GetInputOperands().size(); ++l) {
                        if (op2->GetInputOperands()[l] == aliasIn0) {
                            op2->GetInputOperands()[l] = out0;
                            aliasIn0->RemoveConsumer(op2);
                            out0->AddConsumer(op2);
                        }
                    }

                    for (const auto& x: op2->GetOutputOperands()) {
                        if (x->GetParams().find("__alias__") != x->GetParams().end() &&
                            x->GetParams().at("__alias__")->toValue<int>() == aliasIdx) {
                            x->GetParams()["__alias__"] = std::make_shared<Parameter>(newAliasIdx);
                        }
                    }
                }

                // rewind to the updated copy op
                j -= chainsx_op_indexes.size();
            }
        }
    }

    // step9: clear all alias tag
    {
        for (const auto& x: graph.GetOperands()) {
            x->GetParams().erase("__alias__");
        }
    }
}

}// namespace pnnx
