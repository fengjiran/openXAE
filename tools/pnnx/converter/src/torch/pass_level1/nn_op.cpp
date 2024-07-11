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
        // Get adaptive avg pool1d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_avg_pool1d");
        auto p = CreateParameterFromTorchValue(node->namedInput("output_size"));
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveAvgPool1d);

class AdaptiveAvgPool2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d";
    }

    std::string TypeStr() const override {
        return "nn.AdaptiveAvgPool2d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive avg pool2d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_avg_pool2d");
        auto p = CreateParameterFromTorchValue(node->namedInput("output_size"));
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveAvgPool2d);

class AdaptiveAvgPool3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool3d";
    }

    std::string TypeStr() const override {
        return "nn.AdaptiveAvgPool3d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive avg pool3d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_avg_pool3d");
        auto p = CreateParameterFromTorchValue(node->namedInput("output_size"));
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveAvgPool3d);

class AdaptiveMaxPool1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AdaptiveMaxPool1d";
    }

    std::string TypeStr() const override {
        return "nn.AdaptiveMaxPool1d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive max pool1d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_max_pool1d");
        auto p1 = CreateParameterFromTorchValue(node->namedInput("output_size"));
        auto p2 = graph->outputs()[0]->node()->kind() == c10::prim::TupleConstruct ? true : false;

        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p1);
        op->GetParameters()["return_indices"] = std::make_shared<Parameter>(p2);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveMaxPool1d);

class AdaptiveMaxPool2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AdaptiveMaxPool2d";
    }

    std::string TypeStr() const override {
        return "nn.AdaptiveMaxPool2d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive max pool2d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_max_pool2d");
        auto p1 = CreateParameterFromTorchValue(node->namedInput("output_size"));
        auto p2 = graph->outputs()[0]->node()->kind() == c10::prim::TupleConstruct ? true : false;

        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p1);
        op->GetParameters()["return_indices"] = std::make_shared<Parameter>(p2);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveMaxPool2d);

class AdaptiveMaxPool3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AdaptiveMaxPool3d";
    }

    std::string TypeStr() const override {
        return "nn.AdaptiveMaxPool3d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive max pool3d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_max_pool3d");
        auto p1 = CreateParameterFromTorchValue(node->namedInput("output_size"));
        auto p2 = graph->outputs()[0]->node()->kind() == c10::prim::TupleConstruct ? true : false;

        op->GetParameters()["output_size"] = std::make_shared<Parameter>(p1);
        op->GetParameters()["return_indices"] = std::make_shared<Parameter>(p2);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AdaptiveMaxPool3d);

class AlphaDropout : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.dropout.AlphaDropout";
    }

    std::string TypeStr() const override {
        return "nn.AlphaDropout";
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AlphaDropout);

class AvgPool1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AvgPool1d";
    }

    std::string TypeStr() const override {
        return "nn.AvgPool1d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get avg pool1d node
        const auto* node = FindNodeByKind(graph, "aten::avg_pool1d");
        auto p1 = CreateParameterFromTorchValue(node->namedInput("kernel_size"));
        auto p2 = CreateParameterFromTorchValue(node->namedInput("stride"));
        auto p3 = CreateParameterFromTorchValue(node->namedInput("padding"));
        auto p4 = CreateParameterFromTorchValue(node->namedInput("ceil_mode"));
        auto p5 = CreateParameterFromTorchValue(node->namedInput("count_include_pad"));

        op->GetParameters()["kernel_size"] = std::make_shared<Parameter>(p1);
        op->GetParameters()["stride"] = std::make_shared<Parameter>(p2);
        op->GetParameters()["padding"] = std::make_shared<Parameter>(p3);
        op->GetParameters()["ceil_mode"] = std::make_shared<Parameter>(p4);
        op->GetParameters()["count_include_pad"] = std::make_shared<Parameter>(p5);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AvgPool1d);

class AvgPool2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AvgPool2d";
    }

    std::string TypeStr() const override {
        return "nn.AvgPool2d";
    }

    void Write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* node = FindNodeByKind(graph, "aten::avg_pool2d");
        auto p1 = CreateParameterFromTorchValue(node->namedInput("kernel_size"));
        auto p2 = CreateParameterFromTorchValue(node->namedInput("stride"));
        auto p3 = CreateParameterFromTorchValue(node->namedInput("padding"));
        auto p4 = CreateParameterFromTorchValue(node->namedInput("ceil_mode"));
        auto p5 = CreateParameterFromTorchValue(node->namedInput("count_include_pad"));
        auto p6 = CreateParameterFromTorchValue(node->namedInput("divisor_override"));

        op->GetParameters()["kernel_size"] = std::make_shared<Parameter>(p1);
        op->GetParameters()["stride"] = std::make_shared<Parameter>(p2);
        op->GetParameters()["padding"] = std::make_shared<Parameter>(p3);
        op->GetParameters()["ceil_mode"] = std::make_shared<Parameter>(p4);
        op->GetParameters()["count_include_pad"] = std::make_shared<Parameter>(p5);
        op->GetParameters()["divisor_override"] = std::make_shared<Parameter>(p6);
    }
};

REGISTER_PNNX_FUSE_MODULE_PASS(AvgPool2d);

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
