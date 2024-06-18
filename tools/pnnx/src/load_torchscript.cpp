//
// Created by richard on 6/15/24.
//

#include "load_torchscript.h"

//#include <dlfcn.h>
//#include <torch/csrc/api/include/torch/version.h>
#include <torch/script.h>

namespace pnnx {

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

#if Torch_VERSION_MAJOR >= 2 || (Torch_VERSION_MAJOR >= 1 && Torch_VERSION_MINOR >= 9)
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
                        //                        p = {};
                        std::cerr << "Unknown Parameter value kind " << value_node->kind().toDisplayString()
                                  << " of TensorType, t.dim = 0\n";
                    }
                } else {
                    // constant tensor will become pnnx attribute node later.
                    //                    p = {};
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
                        //                        p = {};
                        std::cerr << "Unknown Parameter value list element kind "
                                  << c10::typeKindToString(value_node->output()->type()->containedTypes()[0]->kind())
                                  << std::endl;
                        break;
                    }
                }
                break;
            }

            default: {
                //                p = {};
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
#if Torch_VERSION_MAJOR >= 2 || (Torch_VERSION_MAJOR >= 1 && Torch_VERSION_MINOR >= 9)
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
                //                p = {};
                std::cerr << "Unknown Parameter value list element kind "
                          << c10::typeKindToString(value_node->output()->type()->cast<c10::ListType>()->getElementType()->kind())
                          << std::endl;
                break;
            }
        }
    } else {
        //        p = {};
        std::cerr << "Unknown Parameter value_node kind "
                  << value_node->kind().toDisplayString()
                  << std::endl;
    }

    return p;
}

ParameterVar CreateParameterFromTorchValue(const torch::jit::Value* value) {
    return CreateParameterFromTorchNode(value->node());
}

int load_torchscript(const std::string& ptpath,
                     Graph& pnnx_graph,
                     const std::string& device) {
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

    return 0;
}

}// namespace pnnx