//
// Created by richard on 7/10/24.
//

#include "torch/pass_level1.h"
#include "torch/torch_optimization.h"

namespace pnnx {

class AdaptiveAvgPool1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool1d";
    }

    std::string TypeStr() const override {
        return "nn.AdaptiveAvgPool1d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* adaptive_avg_pool1d = FindNodeByKind(graph, "aten::adaptive_avg_pool1d");
        auto p = CreateParameterFromTorchValue(adaptive_avg_pool1d->namedInput("output_size"));
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveAvgPool1d);

class ReLU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.ReLU";
    }

    std::string TypeStr() const override {
        return "nn.ReLU";
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(ReLU);

}// namespace pnnx
