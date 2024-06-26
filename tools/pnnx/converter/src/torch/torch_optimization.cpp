//
// Created by richard on 6/22/24.
//

#include "torch_optimization.h"

#include <dlfcn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/normalize_ops.h>

//#include <torch/csrc/jit/api/module.h>
//#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

namespace pnnx {

std::shared_ptr<torch::jit::Graph> OptimizeTorchScript(torch::jit::Module& mod) {
    mod.eval();
//    mod = torch::jit::freeze_module(mod);
    auto method = mod.find_method("forward");
    if (!method) {
        auto methods = mod.get_methods();
        if (methods.empty()) {
            std::cerr << "No method in torchscript.\n";
            return {};
        }
        method = methods[0];
        std::cerr << "Use method " << method->name() << " as the entrypoint instead of forward.\n";
    }
    auto graph = method->graph();

    std::cout << "Before Inline:\n";
    graph->dump();
        torch::jit::Inline(*graph);
//    inline_block(graph, {});
//    InlineBlock(graph->block());
    std::cout << "After Inline:\n";
    graph->dump();
    //    torch::jit::NormalizeOps(graph);

    return graph;
}

static DataType GetATTensorType(const at::ScalarType& st) {
    if (st == c10::ScalarType::Float) return DataType::kDataTypeFloat32;
    if (st == c10::ScalarType::Double) return DataType::kDataTypeFloat64;
    if (st == c10::ScalarType::Half) return DataType::kDataTypeFloat16;
    if (st == c10::ScalarType::Int) return DataType::kDataTypeInt32;
    if (st == c10::ScalarType::QInt32) return DataType::kDataTypeInt32;
    if (st == c10::ScalarType::Long) return DataType::kDataTypeInt64;
    if (st == c10::ScalarType::Short) return DataType::kDataTypeInt16;
    if (st == c10::ScalarType::Char) return DataType::kDataTypeInt8;
    if (st == c10::ScalarType::QInt8) return DataType::kDataTypeInt8;
    if (st == c10::ScalarType::Byte) return DataType::kDataTypeUInt8;
    if (st == c10::ScalarType::QUInt8) return DataType::kDataTypeUInt8;
    if (st == c10::ScalarType::Bool) return DataType::kDataTypeBool;
    if (st == c10::ScalarType::ComplexFloat) return DataType::kDataTypeComplex64;
    if (st == c10::ScalarType::ComplexDouble) return DataType::kDataTypeComplex128;
    if (st == c10::ScalarType::ComplexHalf) return DataType::kDataTypeComplex32;
    if (st == c10::ScalarType::BFloat16) return DataType::kDataTypeBFloat16;
    return DataType::kDataTypeUnknown;
}

static c10::ScalarType InputType2C10ScalarType(const std::string& t) {
    if (t == "c64") return torch::kComplexFloat;
    if (t == "c32") return torch::kComplexHalf;
    if (t == "c128") return torch::kComplexDouble;
    if (t == "bf16") return torch::kBFloat16;
    if (t == "f32") return torch::kFloat32;
    if (t == "f16") return torch::kFloat16;
    if (t == "f64") return torch::kFloat64;
    if (t == "i32") return torch::kInt32;
    if (t == "i16") return torch::kInt16;
    if (t == "i64") return torch::kInt64;
    if (t == "i8") return torch::kInt8;
    if (t == "u8") return torch::kUInt8;

    std::cerr << "Unsupported type " << t << " fallback to float32.\n";
    return torch::kFloat32;
}

ParameterVar CreateParameterFromTorchNode(const torch::jit::Node* value_node) {
    ParameterVar p;
    if (value_node->kind() == c10::prim::Constant) {
        if (value_node->output()->type()->kind() == c10::TypeKind::NoneType) {
            return p;
        }

        if (!value_node->hasAttribute(torch::jit::attr::value)) {
            std::cerr << "No attribute value.\n";
            value_node->dump();
            return p;
        }

        switch (value_node->output()->type()->kind()) {
            case c10::TypeKind::BoolType: {
                p = Parameter((bool) value_node->i(torch::jit::attr::value));
                break;
            }

            case c10::TypeKind::IntType: {
                int64_t i64 = value_node->i(torch::jit::attr::value);
                if (i64 == std::numeric_limits<int64_t>::max()) {
                    i64 = std::numeric_limits<int>::max();
                }

                if (i64 == std::numeric_limits<int64_t>::min()) {
                    i64 = std::numeric_limits<int>::min();
                }

                p = Parameter((int) i64);
                break;
            }

            case c10::TypeKind::FloatType: {
                p = Parameter((float) value_node->f(torch::jit::attr::value));
                break;
            }

            case c10::TypeKind::StringType:
            case c10::TypeKind::DeviceObjType: {
                p = Parameter(std::string(value_node->s(torch::jit::attr::value)));
                break;
            }

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
            case c10::TypeKind::ComplexType: {
                p = Parameter(std::complex<float>(value_node->c(torch::jit::attr::value)));
                break;
            }
#endif

            case c10::TypeKind::TensorType: {
                const at::Tensor& t = value_node->t(torch::jit::attr::value);
                if (t.dim() == 0 && t.numel() == 1) {
                    if (t.scalar_type() == c10::ScalarType::Long) {
                        int64_t i64 = value_node->i(torch::jit::attr::value);
                        if (i64 == std::numeric_limits<int64_t>::max()) {
                            i64 = std::numeric_limits<int>::max();
                        }

                        if (i64 == std::numeric_limits<int64_t>::min()) {
                            i64 = std::numeric_limits<int>::min();
                        }

                        p = Parameter((int) i64);
                    } else if (t.scalar_type() == c10::ScalarType::Int) {
                        p = Parameter(t.item<int>());
                    } else if (t.scalar_type() == c10::ScalarType::Double) {
                        p = Parameter((float) t.item<double>());
                    } else if (t.scalar_type() == c10::ScalarType::Float) {
                        p = Parameter(t.item<float>());
                    } else if (t.scalar_type() == c10::ScalarType::ComplexDouble) {
                        p = Parameter(std::complex<float>(t.item<c10::complex<double>>()));
                    } else if (t.scalar_type() == c10::ScalarType::ComplexFloat) {
                        p = Parameter(std::complex<float>(t.item<c10::complex<float>>()));
                    } else {
                        std::cerr << "Unknown Parameter value kind "
                                  << value_node->kind().toDisplayString()
                                  << " of TensorType, t.dim = 0\n";
                    }
                } else {
                    // constant tensor will become pnnx attribute node later.
                    std::visit([](auto&& arg) { arg.SetType(ParameterType::kParameterOther); }, p);
                }
                break;
            }

            case c10::TypeKind::ListType: {
                switch (value_node->output()->type()->containedTypes()[0]->kind()) {
                    case c10::TypeKind::IntType: {
                        std::vector<int64_t> i64s = value_node->ival(torch::jit::attr::value).toIntVector();
                        std::vector<int> i32s;
                        i32s.reserve(i64s.size());
                        for (auto& i64: i64s) {
                            if (i64 == std::numeric_limits<int64_t>::max()) {
                                i64 = std::numeric_limits<int>::max();
                            }

                            if (i64 == std::numeric_limits<int64_t>::min()) {
                                i64 = std::numeric_limits<int>::min();
                            }
                            i32s.push_back((int) i64);
                        }
                        p = Parameter(i32s);
                        break;
                    }

                    case c10::TypeKind::FloatType: {
                        std::vector<double> doubles = value_node->ival(torch::jit::attr::value).toDoubleVector();
                        std::vector<float> floats;
                        floats.reserve(doubles.size());
                        for (const auto& t: doubles) {
                            floats.push_back((float) t);
                        }
                        p = Parameter(floats);
                        break;
                    }

                    default: {
                        std::cerr << "Unknown Parameter value list element kind "
                                  << c10::typeKindToString(value_node->output()->type()->containedTypes()[0]->kind())
                                  << std::endl;
                        break;
                    }
                }
                break;
            }

            default: {
                std::cerr << "Unknown Parameter value kind "
                          << c10::typeKindToString(value_node->output()->type()->kind())
                          << std::endl;
            }
        }
    } else if (value_node->kind() == c10::prim::ListConstruct) {
        switch (value_node->output()->type()->cast<c10::ListType>()->getElementType()->kind()) {
            case c10::TypeKind::IntType: {
                std::vector<int> ai;
                for (const auto& x: value_node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in int list\n";
                        ai.push_back(0);
                        continue;
                    }
                    ai.push_back((int) x->node()->i(torch::jit::attr::value));
                }
                p = Parameter(ai);
                break;
            }

            case c10::TypeKind::FloatType: {
                std::vector<float> af;
                for (const auto& x: value_node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in float list\n";
                        af.push_back(0);
                        continue;
                    }
                    af.push_back((float) x->node()->f(torch::jit::attr::value));
                }
                p = Parameter(af);
                break;
            }

            case c10::TypeKind::StringType: {
                std::vector<std::string> as;
                for (const auto& x: value_node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in string list\n";
                        as.emplace_back("");
                        continue;
                    }
                    as.emplace_back(x->node()->s(torch::jit::attr::value));
                }
                p = Parameter(as);
                break;
            }
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
            case c10::TypeKind::ComplexType: {
                std::vector<std::complex<float>> ac;
                for (const auto& x: value_node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in complex list\n";
                        ac.emplace_back(0, 0);
                        continue;
                    }
                    ac.emplace_back(std::complex<float>(x->node()->c(torch::jit::attr::value)));
                }
                p = Parameter(ac);
                break;
            }
#endif
            default: {
                std::cerr << "Unknown Parameter value list element kind "
                          << c10::typeKindToString(value_node->output()->type()->cast<c10::ListType>()->getElementType()->kind())
                          << std::endl;
                break;
            }
        }
    } else {
        std::cerr << "Unknown Parameter value_node kind "
                  << value_node->kind().toDisplayString()
                  << std::endl;
    }

    return p;
}

ParameterVar CreateParameterFromTorchValue(const torch::jit::Value* value) {
    return CreateParameterFromTorchNode(value->node());
}

Attribute::Attribute(const at::Tensor& t) {
    type_ = GetATTensorType(t.scalar_type());
    const int ndim = static_cast<int>(t.dim());
    if (ndim == 0) {
        shape_ = {1};
        data_.resize(GetElemSize());
        switch (t.scalar_type()) {
            case c10::ScalarType::Int: {
                int i = t.item<int>();
                memcpy((void*) data_.data(), (const void*) &i, data_.size());
                break;
            }

            case c10::ScalarType::Long: {
                auto i = t.item<int64_t>();
                memcpy((void*) data_.data(), (const void*) &i, data_.size());
                break;
            }

            case c10::ScalarType::Float: {
                auto f = t.item<float>();
                memcpy((void*) data_.data(), (const void*) &f, data_.size());
                break;
            }

            case c10::ScalarType::Double: {
                auto f = t.item<double>();
                memcpy((void*) data_.data(), (const void*) &f, data_.size());
                break;
            }

            default: {
                std::cerr << "Unknown Attribute tensor scalar type "
                          << (int) type_ << std::endl;
            }
        }
    } else {
        shape_.resize(ndim);
        for (int i = 0; i < ndim; ++i) {
            shape_[i] = static_cast<int>(t.size(i));
        }

        if (!shape_.empty()) {
            data_.resize(size() * GetElemSize());
            memcpy((void*) data_.data(), (const void*) t.cpu().contiguous().data_ptr(), data_.size());
        }
    }
}

std::shared_ptr<Operand> Graph::CreateOperand(const torch::jit::Value* value) {
    auto r = CreateOperand(value->debugName());
    r->SetType(DataType::kDataTypeUnknown);
    auto pt = value->type()->cast<c10::TensorType>();
    if (pt) {
        if (pt->scalarType().has_value() && pt->dim().has_value()) {
            r->SetType(GetATTensorType(pt->scalarType().value()));
            const int ndim = (int) pt->dim().value();
            r->GetShape().resize(ndim);
            for (int i = 0; i < ndim; ++i) {
                if (pt->sizes()[i].has_value()) {
                    r->GetShape()[i] = (int) pt->sizes()[i].value();
                } else {
                    r->GetShape()[i] = -1;
                }
            }
        }
    }

    return r;
}

const torch::jit::Node* FindNodeByKind(const std::shared_ptr<torch::jit::Graph>& graph,
                                       const std::string& kind) {
    for (const auto& n: graph->nodes()) {
        if (n->kind().toDisplayString() == kind) {
            return n;
        }
    }

    return nullptr;
}


int load_torchscript(const std::string& ptpath,
                     Graph& pnnx_graph,
                     const std::string& device,
                     const std::vector<std::vector<int64_t>>& input_shapes,
                     const std::vector<std::string>& input_types,
                     const std::vector<std::vector<int64_t>>& input_shapes2,
                     const std::vector<std::string>& input_types2) {
    std::vector<at::Tensor> input_tensors;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        const std::vector<int64_t>& shape = input_shapes[i];
        const std::string& type = input_types[i];
        at::Tensor t = torch::ones(shape, InputType2C10ScalarType(type));
        if (device == "gpu") {
            t = t.cuda();
        }
        input_tensors.push_back(t);
    }

    std::vector<at::Tensor> input_tensors2;
    for (size_t i = 0; i < input_shapes2.size(); ++i) {
        const std::vector<int64_t>& shape = input_shapes2[i];
        const std::string& type = input_types2[i];
        at::Tensor t = torch::ones(shape, InputType2C10ScalarType(type));
        if (device == "gpu") {
            t = t.cuda();
        }
        input_tensors2.push_back(t);
    }


    torch::jit::Module mod;
    try {
        mod = torch::jit::load(ptpath, (device == "gpu") ? c10::kCUDA : c10::kCPU);
    } catch (const c10::Error& e) {
        std::cerr << "Load torchscript failed: " << e.what() << std::endl;
        std::cerr << "Please export model to torchscript as follows:\n";
        std::cerr << "------------------------------------------\n";
        std::cerr << "import torch\n";
        std::cerr << "import torchvision.models as models\n\n";
        std::cerr << "net = models.resnet18(pretrained=True)\n";
        std::cerr << "net = net.eval()\n\n";
        std::cerr << "x = torch.rand(1, 3, 224, 224)\n";
        std::cerr << "mod = torch.jit.trace(net, x)\n";
        std::cerr << "mod.save(\"resnet18.pt\")\n";
        std::cerr << "------------------------------------------\n";

        return -1;
    }

    mod.eval();

    auto method = mod.find_method("forward");
    if (!method) {
        auto methods = mod.get_methods();
        if (methods.empty()) {
            std::cerr << "No method in torchscript.\n";
            return -1;
        }
        method = methods[0];
        std::cerr << "Use method " << method->name() << " as the entrypoint instead of forward.\n";
    }

    auto g = method->graph();

    g->dump();

    std::cerr << "############# pass_level0\n";


    return 0;
}

}// namespace pnnx
