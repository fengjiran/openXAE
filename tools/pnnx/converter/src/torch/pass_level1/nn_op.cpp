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

class Fold : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.fold.Fold";
    }

    std::string TypeStr() const override {
        return "nn.Fold";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* col2im = FindNodeByKind(graph, "aten::col2im");
        auto& params = op->GetParameters();

        params["output_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(col2im->namedInput("output_size")));
        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(col2im->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(col2im->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(col2im->namedInput("padding")));
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(col2im->namedInput("dilation")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Fold);

class GELU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.GELU";
    }

    std::string TypeStr() const override {
        return "nn.GELU";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(GELU);

class GLU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.GLU";
    }

    std::string TypeStr() const override {
        return "nn.GLU";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* glu = FindNodeByKind(graph, "aten::glu");

        op->GetParameters()["dim"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(glu->namedInput("dim")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(GLU);

class GroupNorm : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.normalization.GroupNorm";
    }

    std::string TypeStr() const override {
        return "nn.GroupNorm";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* gn = FindNodeByKind(graph, "aten::group_norm");
        auto& params = op->GetParameters();

        params["num_groups"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(gn->namedInput("num_groups")));
        params["eps"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(gn->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));

        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            const auto& weight = mod.attr("weight").toTensor();

            params["num_channels"] = std::make_shared<Parameter>(weight.size(0));

            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        } else {
            std::cerr << "Cannot resolve GroupNorm num_channels when affine=False\n";
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(GroupNorm);

class GRU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.rnn.GRU";
    }

    std::string TypeStr() const override {
        return "nn.GRU";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* gru = FindNodeByKind(graph, "aten::gru");
        const auto* return_tuple = FindNodeByKind(graph, "prim::TupleConstruct");
        auto& params = op->GetParameters();
        if (return_tuple && return_tuple->inputs().size() == 2 && gru->outputs().size() == 2 && return_tuple->inputs()[0] == gru->outputs()[1] && return_tuple->inputs()[1] == gru->outputs()[0]) {
            // mark the swapped output tuple
            // we would restore the fine order in pass_level3/fuse_rnn_unpack
            std::cerr << "swapped detected !\n";
            params["pnnx_rnn_output_swapped"] = std::make_shared<Parameter>(1);
        }

        const auto& weight_ih_l0 = mod.attr("weight_ih_l0").toTensor();

        params["input_size"] = std::make_shared<Parameter>(weight_ih_l0.size(1));
        params["hidden_size"] = std::make_shared<Parameter>(weight_ih_l0.size(0) / 3);
        params["num_layers"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(gru->namedInput("num_layers")));
        params["bias"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(gru->namedInput("has_biases")));
        params["batch_first"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(gru->namedInput("batch_first")));
        params["bidirectional"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(gru->namedInput("bidirectional")));

        const int num_layers = params["num_layers"]->toValue<int>();
        const bool bias = params["bias"]->toValue<bool>();
        const bool bidirectional = params["bidirectional"]->toValue<bool>();

        for (int k = 0; k < num_layers; k++) {
            std::string weight_ih_lk_key = std::string("weight_ih_l") + std::to_string(k);
            std::string weight_hh_lk_key = std::string("weight_hh_l") + std::to_string(k);

            op->GetAttributes()[weight_ih_lk_key] = std::make_shared<Attribute>(mod.attr(weight_ih_lk_key).toTensor());
            op->GetAttributes()[weight_hh_lk_key] = std::make_shared<Attribute>(mod.attr(weight_hh_lk_key).toTensor());

            if (bias) {
                std::string bias_ih_lk_key = std::string("bias_ih_l") + std::to_string(k);
                std::string bias_hh_lk_key = std::string("bias_hh_l") + std::to_string(k);

                op->GetAttributes()[bias_ih_lk_key] = std::make_shared<Attribute>(mod.attr(bias_ih_lk_key).toTensor());
                op->GetAttributes()[bias_hh_lk_key] = std::make_shared<Attribute>(mod.attr(bias_hh_lk_key).toTensor());
            }

            if (bidirectional) {
                std::string weight_ih_lk_reverse_key = std::string("weight_ih_l") + std::to_string(k) + "_reverse";
                std::string weight_hh_lk_reverse_key = std::string("weight_hh_l") + std::to_string(k) + "_reverse";

                op->GetAttributes()[weight_ih_lk_reverse_key] = std::make_shared<Attribute>(mod.attr(weight_ih_lk_reverse_key).toTensor());
                op->GetAttributes()[weight_hh_lk_reverse_key] = std::make_shared<Attribute>(mod.attr(weight_hh_lk_reverse_key).toTensor());

                if (bias) {
                    std::string bias_ih_lk_reverse_key = std::string("bias_ih_l") + std::to_string(k) + "_reverse";
                    std::string bias_hh_lk_reverse_key = std::string("bias_hh_l") + std::to_string(k) + "_reverse";

                    op->GetAttributes()[bias_ih_lk_reverse_key] = std::make_shared<Attribute>(mod.attr(bias_ih_lk_reverse_key).toTensor());
                    op->GetAttributes()[bias_hh_lk_reverse_key] = std::make_shared<Attribute>(mod.attr(bias_hh_lk_reverse_key).toTensor());
                }
            }
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(GRU);

class Hardshrink : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.Hardshrink";
    }

    std::string TypeStr() const override {
        return "nn.Hardshrink";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* hardshrink = FindNodeByKind(graph, "aten::hardshrink");

        op->GetParameters()["lambd"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(hardshrink->namedInput("lambd")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Hardshrink);

class Hardsigmoid : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.Hardsigmoid";
    }

    std::string TypeStr() const override {
        return "nn.Hardsigmoid";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Hardsigmoid);

class Hardswish : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.Hardswish";
    }

    std::string TypeStr() const override {
        return "nn.Hardswish";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Hardswish);

class Hardtanh : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.Hardtanh";
    }

    std::string TypeStr() const override {
        return "nn.Hardtanh";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* hardtanh = FindNodeByKind(graph, "aten::hardtanh");
        auto& params = op->GetParameters();

        params["min_val"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(hardtanh->namedInput("min_val")));
        params["max_val"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(hardtanh->namedInput("max_val")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Hardtanh);

class InstanceNorm1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.instancenorm.InstanceNorm1d";
    }

    std::string TypeStr() const override {
        return "nn.InstanceNorm1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* in = FindNodeByKind(graph, "aten::instance_norm");
        auto& params = op->GetParameters();

        params["eps"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(in->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));
        params["track_running_stats"] = std::make_shared<Parameter>(mod.hasattr("running_mean") && mod.hasattr("running_var"));

        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            const auto& weight = mod.attr("weight").toTensor();

            params["num_features"] = std::make_shared<Parameter>(weight.size(0));

            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }

        if (mod.hasattr("running_mean") && mod.hasattr("running_var")) {
            const auto& running_mean = mod.attr("running_mean").toTensor();

            params["num_features"] = std::make_shared<Parameter>(running_mean.size(0));

            op->GetAttributes()["running_mean"] = std::make_shared<Attribute>(mod.attr("running_mean").toTensor());
            op->GetAttributes()["running_var"] = std::make_shared<Attribute>(mod.attr("running_var").toTensor());
        }

        // take num_features from input shape
        if (!op->HasParam("num_features") && !op->GetInputOperands()[0]->GetShape().empty()) {
            params["num_features"] = std::make_shared<Parameter>(
                    op->GetInputOperands()[0]->GetShape()[op->GetInputOperands()[0]->GetShape().size() - 2]);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(InstanceNorm1d);

class InstanceNorm2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.instancenorm.InstanceNorm2d";
    }

    std::string TypeStr() const override {
        return "nn.InstanceNorm2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* in = FindNodeByKind(graph, "aten::instance_norm");
        auto& params = op->GetParameters();

        params["eps"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(in->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));
        params["track_running_stats"] = std::make_shared<Parameter>(mod.hasattr("running_mean") && mod.hasattr("running_var"));

        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            const auto& weight = mod.attr("weight").toTensor();

            params["num_features"] = std::make_shared<Parameter>(weight.size(0));

            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }

        if (mod.hasattr("running_mean") && mod.hasattr("running_var")) {
            const auto& running_mean = mod.attr("running_mean").toTensor();

            params["num_features"] = std::make_shared<Parameter>(running_mean.size(0));

            op->GetAttributes()["running_mean"] = std::make_shared<Attribute>(mod.attr("running_mean").toTensor());
            op->GetAttributes()["running_var"] = std::make_shared<Attribute>(mod.attr("running_var").toTensor());
        }

        // take num_features from input shape
        if (!op->HasParam("num_features") && !op->GetInputOperands()[0]->GetShape().empty()) {
            params["num_features"] = std::make_shared<Parameter>(
                    op->GetInputOperands()[0]->GetShape()[op->GetInputOperands()[0]->GetShape().size() - 2]);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(InstanceNorm2d);

class InstanceNorm3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.instancenorm.InstanceNorm3d";
    }

    std::string TypeStr() const override {
        return "nn.InstanceNorm3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* in = FindNodeByKind(graph, "aten::instance_norm");
        auto& params = op->GetParameters();

        params["eps"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(in->namedInput("eps")));
        params["affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));
        params["track_running_stats"] = std::make_shared<Parameter>(mod.hasattr("running_mean") && mod.hasattr("running_var"));

        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            const auto& weight = mod.attr("weight").toTensor();

            params["num_features"] = std::make_shared<Parameter>(weight.size(0));

            op->GetAttributes()["weight"] = std::make_shared<Attribute>(weight);
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }

        if (mod.hasattr("running_mean") && mod.hasattr("running_var")) {
            const auto& running_mean = mod.attr("running_mean").toTensor();

            params["num_features"] = std::make_shared<Parameter>(running_mean.size(0));

            op->GetAttributes()["running_mean"] = std::make_shared<Attribute>(running_mean);
            op->GetAttributes()["running_var"] = std::make_shared<Attribute>(mod.attr("running_var").toTensor());
        }

        // take num_features from input shape
        if (!op->HasParam("num_features") && !op->GetInputOperands()[0]->GetShape().empty()) {
            params["num_features"] = std::make_shared<Parameter>(
                    op->GetInputOperands()[0]->GetShape()[op->GetInputOperands()[0]->GetShape().size() - 2]);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(InstanceNorm3d);

class LayerNorm : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.normalization.LayerNorm";
    }

    std::string TypeStr() const override {
        return "nn.LayerNorm";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* ln = FindNodeByKind(graph, "aten::layer_norm");
        auto& params = op->GetParameters();

        params["normalized_shape"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(ln->namedInput("normalized_shape")));
        params["eps"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(ln->namedInput("eps")));
        params["elementwise_affine"] = std::make_shared<Parameter>(mod.hasattr("weight") && mod.hasattr("bias"));

        if (mod.hasattr("weight") && mod.hasattr("bias")) {
            op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LayerNorm);

class LeakyReLU : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.LeakyReLU";
    }

    std::string TypeStr() const override {
        return "nn.LeakyReLU";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* leaky_relu = FindNodeByKind(graph, "aten::leaky_relu");
        const auto* leaky_relu_ = FindNodeByKind(graph, "aten::leaky_relu_");

        if (leaky_relu_) {
            leaky_relu = leaky_relu_;
        }

        op->GetParameters()["negative_slope"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(leaky_relu->namedInput("negative_slope")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LeakyReLU);

class Linear : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.linear.Linear";
    }

    std::string TypeStr() const override {
        return "nn.Linear";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        const auto* addmm = FindNodeByKind(graph, "aten::addmm");
        auto& params = op->GetParameters();
        const auto& weight = mod.attr("weight").toTensor();

        params["in_features"] = std::make_shared<Parameter>(weight.size(1));
        params["out_features"] = std::make_shared<Parameter>(weight.size(0));
        params["bias"] = std::make_shared<Parameter>(mod.hasattr("bias") && mod.attr("bias").isTensor());

        op->GetAttributes()["weight"] = std::make_shared<Attribute>(mod.attr("weight").toTensor());
        if (mod.hasattr("bias") && mod.attr("bias").isTensor()) {
            op->GetAttributes()["bias"] = std::make_shared<Attribute>(mod.attr("bias").toTensor());
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Linear);

class LocalResponseNorm : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.normalization.LocalResponseNorm";
    }

    std::string TypeStr() const override {
        return "nn.LocalResponseNorm";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* avg_pool = FindNodeByKind(graph, "aten::avg_pool2d");
        const auto* avg_pool3d = FindNodeByKind(graph, "aten::avg_pool3d");
        const auto* pow = FindNodeByKind(graph, "aten::pow");
        const auto* add = pow->inputs()[0]->node();
        const auto* mul = add->inputs()[0]->node();
        auto& params = op->GetParameters();

        if (avg_pool3d) {
            avg_pool = avg_pool3d;
        }

        params["size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(avg_pool->namedInput("kernel_size")->node()->inputs()[0]));
        params["beta"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(pow->inputs()[1]));
        params["k"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(add->inputs()[1]));
        params["alpha"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(mul->inputs()[1]));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LocalResponseNorm);

class LogSigmoid : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.LogSigmoid";
    }

    std::string TypeStr() const override {
        return "nn.LogSigmoid";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LogSigmoid);

class LogSoftmax : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.LogSoftmax";
    }

    std::string TypeStr() const override {
        return "nn.LogSoftmax";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* log_softmax = FindNodeByKind(graph, "aten::log_softmax");

        op->GetParameters()["dim"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(log_softmax->namedInput("dim")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LogSoftmax);

class LPPool1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.LPPool1d";
    }

    std::string TypeStr() const override {
        return "nn.LPPool1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* pow = FindNodeByKind(graph, "aten::pow");
        const auto* avg_pool1d = FindNodeByKind(graph, "aten::avg_pool1d");
        auto& params = op->GetParameters();

        params["norm_type"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(pow->inputs()[1]));
        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(avg_pool1d->namedInput("kernel_size")->node()->inputs()[0]));
        if (avg_pool1d->namedInput("stride")->node()->inputs().empty()) {
            params["stride"] = params["kernel_size"];
        } else {
            params["stride"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(avg_pool1d->namedInput("stride")->node()->inputs()[0]));
        }
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(avg_pool1d->namedInput("ceil_mode")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LPPool1d);

class LPPool2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.LPPool2d";
    }

    std::string TypeStr() const override {
        return "nn.LPPool2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* pow = FindNodeByKind(graph, "aten::pow");
        const auto* avg_pool2d = FindNodeByKind(graph, "aten::avg_pool2d");
        auto& params = op->GetParameters();

        params["norm_type"] = std::make_shared<Parameter>(CreateParameterFromTorchValue(pow->inputs()[1]));
        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(avg_pool2d->namedInput("kernel_size")));
        if (avg_pool2d->namedInput("stride")->node()->inputs().empty()) {
            params["stride"] = params["kernel_size"];
        } else {
            params["stride"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(avg_pool2d->namedInput("stride")));
        }
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(avg_pool2d->namedInput("ceil_mode")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LPPool2d);

class LSTM : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.rnn.LSTM";
    }

    std::string TypeStr() const override {
        return "nn.LSTM";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        // mod.dump(true, true, true);
        //
        // graph->dump();

        const auto* lstm = FindNodeByKind(graph, "aten::lstm");
        const auto* return_tuple = FindNodeByKind(graph, "prim::TupleConstruct");
        auto& params = op->GetParameters();

        if (return_tuple && return_tuple->inputs().size() == 3 && lstm->outputs().size() == 3 && return_tuple->inputs()[0] == lstm->outputs()[1] && return_tuple->inputs()[1] == lstm->outputs()[2] && return_tuple->inputs()[2] == lstm->outputs()[0]) {
            // mark the swapped output tuple
            // we would restore the fine order in pass_level3/fuse_rnn_unpack
            std::cerr << "swapped detected !\n";
            params["pnnx_rnn_output_swapped"] = std::make_shared<Parameter>(1);
        }

        // for (auto aa : lstm->schema().arguments())
        // {
        //     fprintf(stderr, "arg %s\n", aa.name().c_str());
        // }

        const auto& weight_ih_l0 = mod.attr("weight_ih_l0").toTensor();
        const auto& weight_hh_l0 = mod.attr("weight_hh_l0").toTensor();

        params["input_size"] = std::make_shared<Parameter>(weight_ih_l0.size(1));
        params["hidden_size"] = std::make_shared<Parameter>(weight_ih_l0.size(0) / 4);
        params["num_layers"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(lstm->namedInput("num_layers")));
        params["bias"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(lstm->namedInput("has_biases")));
        params["batch_first"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(lstm->namedInput("batch_first")));
        params["bidirectional"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(lstm->namedInput("bidirectional")));
        params["proj_size"] = std::make_shared<Parameter>(
                weight_ih_l0.size(0) / 4 == weight_hh_l0.size(1) ? 0 : weight_hh_l0.size(1));

        const int num_layers = params["num_layers"]->toValue<int>();
        const bool bias = params["bias"]->toValue<bool>();
        const bool bidirectional = params["bidirectional"]->toValue<bool>();
        const int proj_size = params["proj_size"]->toValue<int>();

        for (int k = 0; k < num_layers; k++) {
            std::string weight_ih_lk_key = std::string("weight_ih_l") + std::to_string(k);
            std::string weight_hh_lk_key = std::string("weight_hh_l") + std::to_string(k);

            op->GetAttributes()[weight_ih_lk_key] = std::make_shared<Attribute>(mod.attr(weight_ih_lk_key).toTensor());
            op->GetAttributes()[weight_hh_lk_key] = std::make_shared<Attribute>(mod.attr(weight_hh_lk_key).toTensor());

            if (bias) {
                std::string bias_ih_lk_key = std::string("bias_ih_l") + std::to_string(k);
                std::string bias_hh_lk_key = std::string("bias_hh_l") + std::to_string(k);

                op->GetAttributes()[bias_ih_lk_key] = std::make_shared<Attribute>(mod.attr(bias_ih_lk_key).toTensor());
                op->GetAttributes()[bias_hh_lk_key] = std::make_shared<Attribute>(mod.attr(bias_hh_lk_key).toTensor());
            }

            if (proj_size > 0) {
                std::string weight_hr_lk_key = std::string("weight_hr_l") + std::to_string(k);

                op->GetAttributes()[weight_hr_lk_key] = std::make_shared<Attribute>(mod.attr(weight_hr_lk_key).toTensor());
            }

            if (bidirectional) {
                std::string weight_ih_lk_reverse_key = std::string("weight_ih_l") + std::to_string(k) + "_reverse";
                std::string weight_hh_lk_reverse_key = std::string("weight_hh_l") + std::to_string(k) + "_reverse";

                op->GetAttributes()[weight_ih_lk_reverse_key] = std::make_shared<Attribute>(
                        mod.attr(weight_ih_lk_reverse_key).toTensor());
                op->GetAttributes()[weight_hh_lk_reverse_key] = std::make_shared<Attribute>(
                        mod.attr(weight_hh_lk_reverse_key).toTensor());

                if (bias) {
                    std::string bias_ih_lk_reverse_key = std::string("bias_ih_l") + std::to_string(k) + "_reverse";
                    std::string bias_hh_lk_reverse_key = std::string("bias_hh_l") + std::to_string(k) + "_reverse";

                    op->GetAttributes()[bias_ih_lk_reverse_key] = std::make_shared<Attribute>(
                            mod.attr(bias_ih_lk_reverse_key).toTensor());
                    op->GetAttributes()[bias_hh_lk_reverse_key] = std::make_shared<Attribute>(
                            mod.attr(bias_hh_lk_reverse_key).toTensor());
                }

                if (proj_size > 0) {
                    std::string weight_hr_lk_reverse_key = std::string("weight_hr_l") + std::to_string(k) + "_reverse";

                    op->GetAttributes()[weight_hr_lk_reverse_key] = std::make_shared<Attribute>(
                            mod.attr(weight_hr_lk_reverse_key).toTensor());
                }
            }
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(LSTM);

class MaxPool1d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.MaxPool1d";
    }

    std::string TypeStr() const override {
        return "nn.MaxPool1d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* max_pool1d = FindNodeByKind(graph, "aten::max_pool1d");
        const auto* max_pool1d_with_indices = FindNodeByKind(graph, "aten::max_pool1d_with_indices");
        auto& params = op->GetParameters();

        if (max_pool1d_with_indices) {
            max_pool1d = max_pool1d_with_indices;
        }

        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool1d->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool1d->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool1d->namedInput("padding")));
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool1d->namedInput("dilation")));
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool1d->namedInput("ceil_mode")));
        params["return_indices"] = std::make_shared<Parameter>(max_pool1d_with_indices ? true : false);
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(MaxPool1d);

class MaxPool2d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.MaxPool2d";
    }

    std::string TypeStr() const override {
        return "nn.MaxPool2d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* max_pool2d = FindNodeByKind(graph, "aten::max_pool2d");
        const auto* max_pool2d_with_indices = FindNodeByKind(graph, "aten::max_pool2d_with_indices");
        auto& params = op->GetParameters();

        if (max_pool2d_with_indices) {
            max_pool2d = max_pool2d_with_indices;
        }

        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool2d->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool2d->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool2d->namedInput("padding")));
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool2d->namedInput("dilation")));
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool2d->namedInput("ceil_mode")));
        params["return_indices"] = std::make_shared<Parameter>(max_pool2d_with_indices ? true : false);
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(MaxPool2d);

class MaxPool3d : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pooling.MaxPool3d";
    }

    std::string TypeStr() const override {
        return "nn.MaxPool3d";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const auto* max_pool3d = FindNodeByKind(graph, "aten::max_pool3d");
        const auto* max_pool3d_with_indices = FindNodeByKind(graph, "aten::max_pool3d_with_indices");
        auto& params = op->GetParameters();

        if (max_pool3d_with_indices) {
            max_pool3d = max_pool3d_with_indices;
        }

        params["kernel_size"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool3d->namedInput("kernel_size")));
        params["stride"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool3d->namedInput("stride")));
        params["padding"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool3d->namedInput("padding")));
        params["dilation"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool3d->namedInput("dilation")));
        params["ceil_mode"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(max_pool3d->namedInput("ceil_mode")));
        params["return_indices"] = std::make_shared<Parameter>(max_pool3d_with_indices ? true : false);
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(MaxPool3d);

//class MaxUnpool2d : public FuseModulePass {
//public:
//    std::string MatchTypeStr() const override {
//        return "__torch__.torch.nn.modules.pooling.MaxUnpool2d";
//    }
//
//    std::string TypeStr() const override {
//        return "nn.MaxUnpool2d";
//    }
//
//    void Write(const std::shared_ptr<Operator>& op,
//               const std::shared_ptr<torch::jit::Graph>& graph,
//               const torch::jit::Module& mod) const override {
//                graph->dump();
//
//        {
//                        Graph pnnx_graph;
//
//                        pass_level1(mod, graph, pnnx_graph);
//
//                        fuse_expression(pnnx_graph);
//
//                        Operator* expr_op = pnnx_graph.ops[2];
//
//            if (expr_op->type == "pnnx.Expression") {
//                std::string expr = expr_op->params["expr"].s;
//
//                int stride0;
//                int stride1;
//                int kernel_size0;
//                int kernel_size1;
//                int padding0;
//                int padding1;
//                int nscan = sscanf(expr.c_str(), "(int(sub(add(mul(sub(size(@0,2),1),%d),%d),%d)),int(sub(add(mul(sub(size(@1,3),1),%d),%d),%d)))", &stride0, &kernel_size0, &padding0, &stride1, &kernel_size1, &padding1);
//                if (nscan == 6) {
//                    op->params["kernel_size"] = Parameter{kernel_size0, kernel_size1};
//                    op->params["stride"] = Parameter{stride0, stride1};
//                    op->params["padding"] = Parameter{padding0 / 2, padding1 / 2};
//                }
//            }
//        }
//
//        const torch::jit::Node* max_unpool2d = find_node_by_kind(graph, "aten::max_unpool2d");
//
//        for (auto aa: max_unpool2d->schema().arguments()) {
//            fprintf(stderr, "arg %s\n", aa.name().c_str());
//        }
//    }
//};
//REGISTER_PNNX_FUSE_MODULE_PASS(MaxUnpool2d);

class Mish : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.Mish";
    }

    std::string TypeStr() const override {
        return "nn.Mish";
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(Mish);

class MultiheadAttention : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.activation.MultiheadAttention";
    }

    std::string TypeStr() const override {
        return "nn.MultiheadAttention";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph,
               const torch::jit::Module& mod) const override {
        // mod.dump(false, false, false);
        // graph->dump();

        const auto* multi_head_attention = FindNodeByKind(graph, "aten::_native_multi_head_attention");
        auto& params = op->GetParameters();

        if (multi_head_attention) {
            params["num_heads"] = std::make_shared<Parameter>(
                    CreateParameterFromTorchValue(multi_head_attention->namedInput("num_head")));
            params["batch_first"] = std::make_shared<Parameter>(true);
            params["add_zero_attn"] = std::make_shared<Parameter>(false);

            if (multi_head_attention->hasNamedInput("mask") && multi_head_attention->namedInput("mask") == graph->inputs()[graph->inputs().size() - 1]) {
                size_t input_count = op->GetInputOperands().size();
                op->GetInputNames().resize(input_count);
                op->GetInputNames()[input_count - 1] = "attn_mask";
            }
        } else {
            const auto* div_num_heads = FindNodeByKind(graph, "aten::div");
            const auto* div_num_heads_18 = FindNodeByKind(graph, "aten::floor_divide");
            if (div_num_heads_18) {
                div_num_heads = div_num_heads_18;
            }

            params["num_heads"] = std::make_shared<Parameter>(
                    div_num_heads->input(1)->node()->t(torch::jit::attr::value).item<int64_t>());

            const auto* transpose_batch_seq = FindNodeByKind(graph, "aten::transpose");

            int transpose_dim0 = transpose_batch_seq->input(1)->node()->i(torch::jit::attr::value);
            int transpose_dim1 = transpose_batch_seq->input(2)->node()->i(torch::jit::attr::value);
            if (transpose_dim0 == 1 && transpose_dim1 == 0) {
                params["batch_first"] = std::make_shared<Parameter>(true);
            }

            const auto* add_zero_attn = FindNodeByKind(graph, "aten::zeros");
            if (add_zero_attn) {
                params["add_zero_attn"] = std::make_shared<Parameter>(true);
            } else {
                params["add_zero_attn"] = std::make_shared<Parameter>(false);
            }

            const torch::jit::Node* scaled_dot_product_attention = FindNodeByKind(graph, "aten::scaled_dot_product_attention");
            if (scaled_dot_product_attention) {
                if (scaled_dot_product_attention->input(3)->type()->kind() != c10::TypeKind::NoneType) {
                    size_t input_count = op->GetInputOperands().size();
                    op->GetInputNames().resize(input_count);
                    op->GetInputNames()[input_count - 1] = "attn_mask";
                }
            }

            // find attention mask addition pattern pre torch-2.1
            const auto* has_attn_mask = FindNodeByKind(graph, "aten::baddbmm");
            if (has_attn_mask) {
                size_t input_count = op->GetInputOperands().size();
                op->GetInputNames().resize(input_count);
                op->GetInputNames()[input_count - 1] = "attn_mask";
            }

            // find attention mask addition pattern pre torch-1.12
            // attn = torch.bmm(Q, K)
            // input0 = torch.add_(attn, attn_mask)
            // attn0 = torch.softmax(input0, -1)
            const torch::jit::Node* softmax = FindNodeByKind(graph, "aten::softmax");
            if (softmax) {
                const torch::jit::Node* add_ = softmax->input(0)->node();
                if (add_ && add_->kind().toDisplayString() == std::string("aten::add_")) {
                    const torch::jit::Node* bmm = add_->input(0)->node();
                    if (bmm && bmm->kind().toDisplayString() == std::string("aten::bmm")) {
                        size_t input_count = op->GetInputOperands().size();
                        op->GetInputNames().resize(input_count);
                        op->GetInputNames()[input_count - 1] = "attn_mask";
                    }
                }
            }
        }

        if (mod.hasattr("in_proj_weight")) {
            const auto& in_proj_weight = mod.attr("in_proj_weight").toTensor();

            params["embed_dim"] = std::make_shared<Parameter>(in_proj_weight.size(1));
            params["kdim"] = std::make_shared<Parameter>(in_proj_weight.size(1));
            params["vdim"] = std::make_shared<Parameter>(in_proj_weight.size(1));
            op->GetAttributes()["in_proj_weight"] = std::make_shared<Attribute>(mod.attr("in_proj_weight").toTensor());
        } else {
            const auto& q_proj_weight = mod.attr("q_proj_weight").toTensor();
            const auto& k_proj_weight = mod.attr("k_proj_weight").toTensor();
            const auto& v_proj_weight = mod.attr("v_proj_weight").toTensor();

            params["embed_dim"] = std::make_shared<Parameter>(q_proj_weight.size(1));
            params["kdim"] = std::make_shared<Parameter>(k_proj_weight.size(1));
            params["vdim"] = std::make_shared<Parameter>(v_proj_weight.size(1));
            op->GetAttributes()["q_proj_weight"] = std::make_shared<Attribute>(mod.attr("q_proj_weight").toTensor());
            op->GetAttributes()["k_proj_weight"] = std::make_shared<Attribute>(mod.attr("k_proj_weight").toTensor());
            op->GetAttributes()["v_proj_weight"] = std::make_shared<Attribute>(mod.attr("v_proj_weight").toTensor());
        }

        //        const auto& out_proj_weight = mod.attr("out_proj").toModule().attr("weight").toTensor();

        op->GetAttributes()["out_proj.weight"] = std::make_shared<Attribute>(mod.attr("out_proj").toModule().attr("weight").toTensor());

        if (mod.hasattr("in_proj_bias") && mod.attr("out_proj").toModule().hasattr("bias")) {
            // bias=True
            //            const auto& in_proj_bias = mod.attr("in_proj_bias").toTensor();
            //            const auto& out_proj_bias = mod.attr("out_proj").toModule().attr("bias").toTensor();

            params["bias"] = std::make_shared<Parameter>(true);
            op->GetAttributes()["in_proj_bias"] = std::make_shared<Attribute>(mod.attr("in_proj_bias").toTensor());
            op->GetAttributes()["out_proj.bias"] = std::make_shared<Attribute>(mod.attr("out_proj").toModule().attr("bias").toTensor());
        } else {
            params["bias"] = std::make_shared<Parameter>(false);

            // the output projection bias always there no matter bias is False in pytorch 1.8
            // this behavior changes since https://github.com/pytorch/pytorch/commit/58d1b3639bc07f9519de18e5a18e575f260c7eeb
            if (mod.attr("out_proj").toModule().hasattr("bias")) {
                const auto& out_proj_bias = mod.attr("out_proj").toModule().attr("bias").toTensor();
                op->GetAttributes()["out_proj.bias"] = std::make_shared<Attribute>(mod.attr("out_proj").toModule().attr("bias").toTensor());
            }
        }

        if (mod.hasattr("bias_k") && mod.hasattr("bias_v")) {
            // add_bias_kv=True
            //            const auto& bias_k = mod.attr("bias_k").toTensor();
            //            const auto& bias_v = mod.attr("bias_v").toTensor();

            params["add_bias_kv"] = std::make_shared<Parameter>(true);
            op->GetAttributes()["bias_k"] = std::make_shared<Attribute>(mod.attr("bias_k").toTensor());
            op->GetAttributes()["bias_v"] = std::make_shared<Attribute>(mod.attr("bias_v").toTensor());
        } else {
            params["add_bias_kv"] = std::make_shared<Parameter>(false);
        }
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(MultiheadAttention);

class PixelShuffle : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pixelshuffle.PixelShuffle";
    }

    std::string TypeStr() const override {
        return "nn.PixelShuffle";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* pixel_shuffle = FindNodeByKind(graph, "aten::pixel_shuffle");

        op->GetParameters()["upscale_factor"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pixel_shuffle->namedInput("upscale_factor")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(PixelShuffle);

class PixelUnshuffle : public FuseModulePass {
public:
    std::string MatchTypeStr() const override {
        return "__torch__.torch.nn.modules.pixelshuffle.PixelUnshuffle";
    }

    std::string TypeStr() const override {
        return "nn.PixelUnshuffle";
    }

    void Write(const std::shared_ptr<Operator>& op,
               const std::shared_ptr<torch::jit::Graph>& graph) const override {
        const torch::jit::Node* pixel_unshuffle = FindNodeByKind(graph, "aten::pixel_unshuffle");

        op->GetParameters()["downscale_factor"] = std::make_shared<Parameter>(
                CreateParameterFromTorchValue(pixel_unshuffle->namedInput("downscale_factor")));
    }
};
REGISTER_PNNX_FUSE_MODULE_PASS(PixelUnshuffle);

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
