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

static void functionize(Graph& graph) {

    // step1: create shadow view/slice/select for each consumer.
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

            for (auto & k : op1->GetInputOperands()) {
                if (k == out0) {
                    k = shadowOut;
                }
            }
        }

        out0->GetConsumers().resize(1);
    }
}

}// namespace pnnx
