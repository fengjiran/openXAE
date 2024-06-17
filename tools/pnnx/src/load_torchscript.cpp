//
// Created by richard on 6/15/24.
//

#include "load_torchscript.h"

#include <dlfcn.h>
#include <torch/csrc/api/include/torch/version.h>
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