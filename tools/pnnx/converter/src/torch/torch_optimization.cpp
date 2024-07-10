//
// Created by richard on 6/22/24.
//

#include "torch_optimization.h"

#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/normalize_ops.h>

//#include <torch/csrc/jit/api/module.h>
//#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

namespace pnnx {

std::shared_ptr<torch::jit::Graph> OptimizeTorchScript(torch::jit::Module& mod,
                                                       const std::vector<at::Tensor>& inputTensors,
                                                       const std::vector<at::Tensor>& inputTensors2,
                                                       const std::vector<std::string>& moduleOperators,
                                                       const std::string& ptPath,
                                                       const std::string& device,
                                                       std::set<std::string>& foldableConstants,
                                                       const std::string& foldableConstantsZippath) {
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
    graph->dump();

    std::cerr << "############# pass_level0\n";
    //    torch::jit::Inline(*graph);
    //    inline_block(graph, {});
    Inline(graph, {});

    //    torch::jit::ConstantPooling(graph);
    ConstantUnpooling(graph);
    ResetDevice(graph, device);
    FlattenInput(graph);
    if (!inputTensors.empty()) {
        ShapeInference(mod, graph, inputTensors, inputTensors2, moduleOperators, ptPath,
                       device, foldableConstants, foldableConstantsZippath);
    }
    //    torch::jit::NormalizeOps(graph);

    std::cerr << "After Optimization:\n";

    std::cerr << "############# pass_level1\n";
    graph->dump();
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

Parameter CreateParameterFromTorchNode(const torch::jit::Node* node) {
    Parameter p;
    if (node->kind() == c10::prim::Constant) {
        if (node->output()->type()->kind() == c10::TypeKind::NoneType) {
            return p;
        }

        if (!node->hasAttribute(torch::jit::attr::value)) {
            std::cerr << "No attribute value.\n";
            node->dump();
            return p;
        }

        switch (node->output()->type()->kind()) {
            case c10::TypeKind::BoolType: {
                p = static_cast<bool>(node->i(torch::jit::attr::value));
                break;
            }

            case c10::TypeKind::IntType: {
                int64_t i64 = node->i(torch::jit::attr::value);
                if (i64 == std::numeric_limits<int64_t>::max()) {
                    i64 = std::numeric_limits<int>::max();
                }

                if (i64 == std::numeric_limits<int64_t>::min()) {
                    i64 = std::numeric_limits<int>::min();
                }

                p = static_cast<int>(i64);
                break;
            }

            case c10::TypeKind::FloatType: {
                p = static_cast<float>(node->f(torch::jit::attr::value));
                break;
            }

            case c10::TypeKind::StringType:
            case c10::TypeKind::DeviceObjType: {
                p = node->s(torch::jit::attr::value);
                break;
            }

            case c10::TypeKind::ComplexType: {
                p = std::complex<float>(node->c(torch::jit::attr::value));
                break;
            }

            case c10::TypeKind::TensorType: {
                const at::Tensor& t = node->t(torch::jit::attr::value);
                if (t.dim() == 0 && t.numel() == 1) {
                    if (t.scalar_type() == c10::ScalarType::Long) {
                        auto i64 = t.item<int64_t>();
                        if (i64 == std::numeric_limits<int64_t>::max()) {
                            i64 = std::numeric_limits<int>::max();
                        }

                        if (i64 == std::numeric_limits<int64_t>::min()) {
                            i64 = std::numeric_limits<int>::min();
                        }
                        p = static_cast<int>(i64);
                    } else if (t.scalar_type() == c10::ScalarType::Int) {
                        p = t.item<int>();
                    } else if (t.scalar_type() == c10::ScalarType::Double) {
                        p = static_cast<float>(t.item<double>());
                    } else if (t.scalar_type() == c10::ScalarType::Float) {
                        p = t.item<float>();
                    } else if (t.scalar_type() == c10::ScalarType::ComplexDouble) {
                        p = std::complex<float>(t.item<c10::complex<double>>());
                    } else if (t.scalar_type() == c10::ScalarType::ComplexFloat) {
                        p = std::complex<float>(t.item<c10::complex<float>>());
                    } else {
                        std::cerr << "Unknown Parameter value kind "
                                  << node->kind().toDisplayString()
                                  << " of TensorType, t.dim = 0\n";
                    }
                } else {
                    // constant tensor will become pnnx attribute node later.
                    std::cerr << "constant tensor will become pnnx attribute node later.\n";
                }
                break;
            }

            case c10::TypeKind::ListType: {
                switch (node->output()->type()->containedTypes()[0]->kind()) {
                    case c10::TypeKind::IntType: {
                        std::vector<int64_t> i64s = node->ival(torch::jit::attr::value).toIntVector();
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
                        p = i32s;
                        break;
                    }

                    case c10::TypeKind::FloatType: {
                        std::vector<double> doubles = node->ival(torch::jit::attr::value).toDoubleVector();
                        std::vector<float> floats;
                        floats.reserve(doubles.size());
                        for (const auto& t: doubles) {
                            floats.push_back((float) t);
                        }
                        p = floats;
                        break;
                    }

                    default: {
                        std::cerr << "Unknown value list element kind "
                                  << c10::typeKindToString(node->output()->type()->containedTypes()[0]->kind())
                                  << std::endl;
                        break;
                    }
                }
                break;
            }

            default: {
                std::cerr << "Unknown value kind "
                          << c10::typeKindToString(node->output()->type()->kind())
                          << std::endl;
            }
        }
    } else if (node->kind() == c10::prim::ListConstruct) {
        switch (node->output()->type()->cast<c10::ListType>()->getElementType()->kind()) {
            case c10::TypeKind::IntType: {
                std::vector<int> ai;
                for (const auto& x: node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in int list\n";
                        ai.push_back(0);
                        continue;
                    }
                    ai.push_back((int) x->node()->i(torch::jit::attr::value));
                }
                p = ai;
                break;
            }

            case c10::TypeKind::FloatType: {
                std::vector<float> af;
                for (const auto& x: node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in float list\n";
                        af.push_back(0);
                        continue;
                    }
                    af.push_back((float) x->node()->f(torch::jit::attr::value));
                }
                p = af;
                break;
            }

            case c10::TypeKind::StringType: {
                std::vector<std::string> as;
                for (const auto& x: node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in string list\n";
                        as.emplace_back("");
                        continue;
                    }
                    as.emplace_back(x->node()->s(torch::jit::attr::value));
                }
                p = as;
                break;
            }

            case c10::TypeKind::ComplexType: {
                std::vector<std::complex<float>> ac;
                for (const auto& x: node->inputs()) {
                    if (!x->node()->hasAttribute(torch::jit::attr::value)) {
                        std::cerr << "No attribute value in complex list\n";
                        ac.emplace_back(0, 0);
                        continue;
                    }
                    ac.emplace_back(std::complex<float>(x->node()->c(torch::jit::attr::value)));
                }
                p = ac;
                break;
            }

            default: {
                std::cerr << "Unknown Parameter value list element kind "
                          << c10::typeKindToString(node->output()->type()->cast<c10::ListType>()->getElementType()->kind())
                          << std::endl;
                break;
            }
        }
    } else {
        std::cerr << "Unknown node kind "
                  << node->kind().toDisplayString()
                  << std::endl;
    }
    return p;
}

Parameter CreateParameterFromTorchValue(const torch::jit::Value* value) {
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

}// namespace pnnx
