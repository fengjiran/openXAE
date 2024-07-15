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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive avg pool1d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_avg_pool1d");
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("output_size")));
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive avg pool2d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_avg_pool2d");
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("output_size")));
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive avg pool3d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_avg_pool3d");
        op->GetParameters()["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("output_size")));
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive max pool1d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_max_pool1d");
        auto& params = op->GetParameters();

        params["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("output_size")));
        params["return_indices"] = std::make_shared<Parameter>(
                graph->outputs()[0]->node()->kind() == c10::prim::TupleConstruct ? true : false);
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive max pool2d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_max_pool2d");
        auto& params = op->GetParameters();

        params["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("output_size")));
        params["return_indices"] = std::make_shared<Parameter>(
                graph->outputs()[0]->node()->kind() == c10::prim::TupleConstruct ? true : false);
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get adaptive max pool3d node
        const auto* node = FindNodeByKind(graph, "aten::adaptive_max_pool3d");
        auto& params = op->GetParameters();

        params["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("output_size")));
        params["return_indices"] = std::make_shared<Parameter>(
                graph->outputs()[0]->node()->kind() == c10::prim::TupleConstruct ? true : false);
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        // Get avg pool1d node
        const auto* node = FindNodeByKind(graph, "aten::avg_pool1d");
        auto& params = op->GetParameters();

        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("padding")));
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("ceil_mode")));
        params["count_include_pad"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("count_include_pad")));
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

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* node = FindNodeByKind(graph, "aten::avg_pool2d");
        auto& params = op->GetParameters();

        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("padding")));
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("ceil_mode")));
        params["count_include_pad"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("count_include_pad")));
        params["divisor_override"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("divisor_override")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(AvgPool2d);

class AvgPool3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.AvgPool3d";
    }

    std::string TypeStr() const override {
        return "nn.AvgPool3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* node = FindNodeByKind(graph, "aten::avg_pool3d");
        auto& params = op->GetParameters();

        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("padding")));
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("ceil_mode")));
        params["count_include_pad"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("count_include_pad")));
        params["divisor_override"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("divisor_override")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(AvgPool3d);

class BatchNorm1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.batchnorm.BatchNorm1d";
    }

    std::string TypeStr() const override {
        return "nn.BatchNorm1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* bn = FindNodeByKind(graph, "aten::batch_norm");
        auto& params = op->GetParameters();
        const auto& running_mean = mod.attr("running_mean").toTensor();
        const auto& running_var = mod.attr("running_var").toTensor();

        params["num_features"] = std::make_shared<Parameter>(running_mean.size(0));
        params["eps"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(bn->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));

        op->GetAttributes()["running_mean"] = std::make_shared<Attribute>(running_mean);
        op->GetAttributes()["running_var"] = std::make_shared<Attribute>(running_var);
        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(BatchNorm1d);

class BatchNorm2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.batchnorm.BatchNorm2d";
    }

    std::string TypeStr() const override {
        return "nn.BatchNorm2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* bn = FindNodeByKind(graph, "aten::batch_norm");
        auto& params = op->GetParameters();
        const auto& running_mean = mod.attr("running_mean").toTensor();
        const auto& running_var = mod.attr("running_var").toTensor();

        params["num_features"] = std::make_shared<Parameter>(running_mean.size(0));
        params["eps"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(bn->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));

        op->GetAttributes()["running_mean"] = std::make_shared<Attribute>(running_mean);
        op->GetAttributes()["running_var"] = std::make_shared<Attribute>(running_var);
        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(BatchNorm2d);

class BatchNorm3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.batchnorm.BatchNorm3d";
    }

    std::string TypeStr() const override {
        return "nn.BatchNorm3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* bn = FindNodeByKind(graph, "aten::batch_norm");
        const auto& running_mean = mod.attr("running_mean").toTensor();
        const auto& running_var = mod.attr("running_var").toTensor();
        auto& params = op->GetParameters();

        params["num_features"] = std::make_shared<Parameter>(running_mean.size(0));
        params["eps"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(bn->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));

        op->GetAttributes()["running_mean"] = std::make_shared<Attribute>(running_mean);
        op->GetAttributes()["running_var"] = std::make_shared<Attribute>(running_var);
        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(BatchNorm3d);

class CELU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.CELU";
    }

    std::string TypeStr() const override {
        return "nn.CELU";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* celu = FindNodeByKind(graph, "aten::celu");

        op->GetParameters()["alpha"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(celu->namedInput("alpha")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(CELU);

class ChannelShuffle : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.channelshuffle.ChannelShuffle";
    }

    std::string TypeStr() const override {
        return "nn.ChannelShuffle";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* node = FindNodeByKind(graph, "aten::channel_shuffle");

        op->GetParameters()["groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(node->namedInput("groups")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ChannelShuffle);

class ConstantPad1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.padding.ConstantPad1d";
    }

    std::string TypeStr() const override {
        return "nn.ConstantPad1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* pad = FindNodeByKind(graph, "aten::pad");
        const torch::jit::Node* constant_pad_nd = FindNodeByKind(graph, "aten::constant_pad_nd");
        auto& params = op->GetParameters();
        if (!pad) {
            pad = constant_pad_nd;
        }

        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pad->namedInput("pad")));
        params["value"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pad->namedInput("value")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ConstantPad1d);

class ConstantPad2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.padding.ConstantPad2d";
    }

    std::string TypeStr() const override {
        return "nn.ConstantPad2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* pad = FindNodeByKind(graph, "aten::pad");
        const torch::jit::Node* constant_pad_nd = FindNodeByKind(graph, "aten::constant_pad_nd");
        auto& params = op->GetParameters();
        if (!pad) {
            pad = constant_pad_nd;
        }

        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pad->namedInput("pad")));
        params["value"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pad->namedInput("value")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ConstantPad2d);

class ConstantPad3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.padding.ConstantPad3d";
    }

    std::string TypeStr() const override {
        return "nn.ConstantPad3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* pad = FindNodeByKind(graph, "aten::pad");
        const torch::jit::Node* constant_pad_nd = FindNodeByKind(graph, "aten::constant_pad_nd");
        auto& params = op->GetParameters();
        if (!pad) {
            pad = constant_pad_nd;
        }

        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pad->namedInput("pad")));
        params["value"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pad->namedInput("value")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ConstantPad3d);

class Conv1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.conv.Conv1d";
    }

    std::string TypeStr() const override {
        return "nn.Conv1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* convolution = FindNodeByKind(graph, "aten::_convolution");
        const auto* convolution_mode = FindNodeByKind(graph, "aten::_convolution_mode");
        const auto* pad = FindNodeByKind(graph, "aten::pad");
        const auto* reflection_pad1d = FindNodeByKind(graph, "aten::reflection_pad1d");
        const auto* replication_pad1d = FindNodeByKind(graph, "aten::replication_pad1d");
        auto& params = op->GetParameters();
        if (convolution_mode) {
            convolution = convolution_mode;
        }

        const auto& weight = mod.attr("weight").toTensor();

        params["groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("groups")));
        const auto& groups = params["groups"]->toValue<int>();
        params["in_channels"] = std::make_shared<Parameter>(weight.size(1) * groups);
        params["out_channels"] = std::make_shared<Parameter>(weight.size(0));
        params["kernel_size"] = std::make_shared<Parameter>(Parameter({weight.size(2)}));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("stride")));

        if (pad) {
            params["padding_mode"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(pad->namedInput("mode")));
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(pad->namedInput("pad")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 2) {
                // Conv1d only accepts tuple of one integer
                if (padding[0] == padding[1]) {
                    padding.resize(1);
                } else if (padding[0] != padding[1]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else if (reflection_pad1d) {
            params["padding_mode"] = std::make_shared<Parameter>("reflect");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(reflection_pad1d->namedInput("padding")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 2) {
                // Conv1d only accepts tuple of one integer
                if (padding[0] == padding[1]) {
                    padding.resize(1);
                } else if (padding[0] != padding[1]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else if (replication_pad1d) {
            params["padding_mode"] = std::make_shared<Parameter>("replicate");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(replication_pad1d->namedInput("padding")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 2) {
                // Conv1d only accepts tuple of one integer
                if (padding[0] == padding[1]) {
                    padding.resize(1);
                } else if (padding[0] != padding[1]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else {
            params["padding_mode"] = std::make_shared<Parameter>("zeros");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(convolution->namedInput("padding")));
        }
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("dilation")));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias"));

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias")) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Conv1d);

class Conv2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.conv.Conv2d";
    }

    std::string TypeStr() const override {
        return "nn.Conv2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* convolution = FindNodeByKind(graph, "aten::_convolution");
        const auto* convolution_mode = FindNodeByKind(graph, "aten::_convolution_mode");
        const auto* pad = FindNodeByKind(graph, "aten::pad");
        const auto* reflection_pad2d = FindNodeByKind(graph, "aten::reflection_pad2d");
        const auto* replication_pad2d = FindNodeByKind(graph, "aten::replication_pad2d");
        auto& params = op->GetParameters();
        if (convolution_mode) {
            convolution = convolution_mode;
        }

        const auto& weight = mod.attr("weight").toTensor();

        params["groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("groups")));
        const auto& groups = params["groups"]->toValue<int>();
        params["in_channels"] = std::make_shared<Parameter>(weight.size(1) * groups);
        params["out_channels"] = std::make_shared<Parameter>(weight.size(0));
        params["kernel_size"] = std::make_shared<Parameter>(Parameter({weight.size(2), weight.size(3)}));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("stride")));

        if (pad) {
            params["padding_mode"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(pad->namedInput("mode")));
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(pad->namedInput("pad")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 4) {
                // Conv2d only accepts tuple of two integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3]) {
                    padding.resize(2);
                } else if (padding[0] == padding[2] && padding[1] == padding[3] && padding[0] != padding[1]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else if (reflection_pad2d) {
            params["padding_mode"] = std::make_shared<Parameter>("reflect");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(reflection_pad2d->namedInput("padding")));

            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 4) {
                // Conv2d only accepts tuple of two integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3]) {
                    padding.resize(2);
                } else if (padding[0] == padding[2] && padding[1] == padding[3] && padding[0] != padding[1]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else if (replication_pad2d) {
            params["padding_mode"] = std::make_shared<Parameter>("replicate");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(replication_pad2d->namedInput("padding")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 4) {
                // Conv2d only accepts tuple of two integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3]) {
                    padding.resize(2);
                } else if (padding[0] == padding[2] && padding[1] == padding[3] && padding[0] != padding[1]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else {
            params["padding_mode"] = std::make_shared<Parameter>("zeros");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(convolution->namedInput("padding")));
        }
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("dilation")));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias") && mod.attr("bias").isTensor());

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias") && mod.attr("bias").isTensor()) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Conv2d);

class Conv3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.conv.Conv3d";
    }

    std::string TypeStr() const override {
        return "nn.Conv3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* convolution = FindNodeByKind(graph, "aten::_convolution");
        const auto* convolution_mode = FindNodeByKind(graph, "aten::_convolution_mode");
        const auto* pad = FindNodeByKind(graph, "aten::pad");
        const auto* reflection_pad3d = FindNodeByKind(graph, "aten::reflection_pad3d");
        const auto* replication_pad3d = FindNodeByKind(graph, "aten::replication_pad3d");
        auto& params = op->GetParameters();
        if (convolution_mode) {
            convolution = convolution_mode;
        }

        const auto& weight = mod.attr("weight").toTensor();

        params["groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("groups")));
        const auto& groups = params["groups"]->toValue<int>();
        params["in_channels"] = std::make_shared<Parameter>(weight.size(1) * groups);
        params["out_channels"] = std::make_shared<Parameter>(weight.size(0));
        params["kernel_size"] = std::make_shared<Parameter>(Parameter({weight.size(2), weight.size(3), weight.size(4)}));
        params["stride"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("stride")));
        if (pad) {
            params["padding_mode"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(pad->namedInput("mode")));
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(pad->namedInput("pad")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 6) {
                // Conv3d only accepts tuple of three integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] && padding[3] == padding[4] && padding[4] == padding[5]) {
                    padding.resize(3);
                } else if (padding[0] == padding[3] && padding[1] == padding[4] && padding[2] == padding[5] && padding[0] != padding[1] && padding[1] != padding[2]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else if (reflection_pad3d) {
            params["padding_mode"] = std::make_shared<Parameter>("reflect");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(reflection_pad3d->namedInput("padding")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 6) {
                // Conv3d only accepts tuple of three integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] && padding[3] == padding[4] && padding[4] == padding[5]) {
                    padding.resize(3);
                } else if (padding[0] == padding[3] && padding[1] == padding[4] && padding[2] == padding[5] && padding[0] != padding[1] && padding[1] != padding[2]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else if (replication_pad3d) {
            params["padding_mode"] = std::make_shared<Parameter>("replicate");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(replication_pad3d->namedInput("padding")));
            auto& padding = params["padding"]->toValue<std::vector<int>>();
            if (padding.size() == 6) {
                // Conv3d only accepts tuple of three integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] && padding[3] == padding[4] && padding[4] == padding[5]) {
                    padding.resize(3);
                } else if (padding[0] == padding[3] && padding[1] == padding[4] && padding[2] == padding[5] && padding[0] != padding[1] && padding[1] != padding[2]) {
                    padding.resize(0);
                    params["padding"] = std::make_shared<Parameter>("same");
                }
            }
        } else {
            params["padding_mode"] = std::make_shared<Parameter>("zeros");
            params["padding"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(convolution->namedInput("padding")));
        }
        params["dilation"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("dilation")));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias"));

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias")) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Conv3d);

class ConvTranspose1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.conv.ConvTranspose1d";
    }

    std::string TypeStr() const override {
        return "nn.ConvTranspose1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const torch::jit::Node* convolution = FindNodeByKind(graph, "aten::_convolution");
        auto& params = op->GetParameters();
        const auto& weight = mod.attr("weight").toTensor();

        params["groups"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("groups")));
        params["in_channels"] = std::make_shared<Parameter>(weight.size(0));
        const auto& groups = params["groups"]->toValue<int>();
        params["out_channels"] = std::make_shared<Parameter>(weight.size(1) * groups);
        params["kernel_size"] = std::make_shared<Parameter>(Parameter({weight.size(2)}));
        params["stride"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("padding")));
        params["output_padding"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("output_padding")));
        params["dilation"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("dilation")));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias"));

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias")) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }

        if (op->GetInputOperands().size() > 1) {
            std::cerr << "ConvTranspose1d arg output_size detected and dropped !\n";

            for (size_t i = 1; i < op->GetInputOperands().size(); i++) {
                op->GetInputOperands()[i]->RemoveConsumer(op);
            }
            op->GetInputOperands().resize(1);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ConvTranspose1d);

class ConvTranspose2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.conv.ConvTranspose2d";
    }

    std::string TypeStr() const override {
        return "nn.ConvTranspose2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* convolution = FindNodeByKind(graph, "aten::_convolution");
        auto& params = op->GetParameters();
        const auto& weight = mod.attr("weight").toTensor();

        params["groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("groups")));
        const auto& groups = params["groups"]->toValue<int>();
        params["in_channels"] = std::make_shared<Parameter>(weight.size(0));
        params["out_channels"] = std::make_shared<Parameter>(weight.size(1) * groups);
        params["kernel_size"] = std::make_shared<Parameter>(Parameter({weight.size(2), weight.size(3)}));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("padding")));
        params["output_padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("output_padding")));
        params["dilation"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(convolution->namedInput("dilation")));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias"));

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias")) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }

        if (op->GetInputOperands().size() > 1) {
            std::cerr << "ConvTranspose2d arg output_size detected and dropped !\n";

            for (size_t i = 1; i < op->GetInputOperands().size(); i++) {
                op->GetInputOperands()[i]->RemoveConsumer(op);
            }
            op->GetInputOperands().resize(1);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ConvTranspose2d);

class ConvTranspose3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.conv.ConvTranspose3d";
    }

    std::string TypeStr() const override {
        return "nn.ConvTranspose3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* convolution = FindNodeByKind(graph, "aten::_convolution");
        auto& params = op->GetParameters();
        const auto& weight = mod.attr("weight").toTensor();

        params["groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("groups")));
        const auto& groups = params["groups"]->toValue<int>();
        params["in_channels"] = std::make_shared<Parameter>(weight.size(0));
        params["out_channels"] = std::make_shared<Parameter>(weight.size(1) * groups);
        params["kernel_size"] = std::make_shared<Parameter>(Parameter({weight.size(2), weight.size(3), weight.size(4)}));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("padding")));
        params["output_padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("output_padding")));
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(convolution->namedInput("dilation")));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias"));

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias")) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }

        if (op->GetInputOperands().size() > 1) {
            std::cerr << "ConvTranspose3d arg output_size detected and dropped !\n";

            for (size_t i = 1; i < op->GetInputOperands().size(); i++) {
                op->GetInputOperands()[i]->RemoveConsumer(op);
            }
            op->GetInputOperands().resize(1);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ConvTranspose3d);

class Dropout : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.dropout.Dropout";
    }

    std::string TypeStr() const override {
        return "nn.Dropout";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Dropout);

class Dropout2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.dropout.Dropout2d";
    }

    std::string TypeStr() const override {
        return "nn.Dropout2d";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Dropout2d);

class Dropout3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.dropout.Dropout3d";
    }

    std::string TypeStr() const override {
        return "nn.Dropout3d";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Dropout3d);

class ELU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.ELU";
    }

    std::string TypeStr() const override {
        return "nn.ELU";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* elu = FindNodeByKind(graph, "aten::elu");

        op->GetParameters()["alpha"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(elu->namedInput("alpha")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(ELU);

class Embedding : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.sparse.Embedding";
    }

    std::string TypeStr() const override {
        return "nn.Embedding";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* embedding = FindNodeByKind(graph, "aten::embedding");
        auto& params = op->GetParameters();
        const auto& weight = mod.attr("weight").toTensor();

        params["num_embeddings"] = std::make_shared<Parameter>(weight.size(0));
        params["embedding_dim"] = std::make_shared<Parameter>(weight.size(1));

        // op->params["padding_idx"] = embedding->namedInput("padding_idx");
        // op->params["scale_grad_by_freq"] = embedding->namedInput("scale_grad_by_freq");
        params["sparse"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(embedding->namedInput("sparse")));

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Embedding);

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
