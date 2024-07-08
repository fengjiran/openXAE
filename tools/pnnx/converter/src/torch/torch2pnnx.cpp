//
// Created by richard on 6/22/24.
//

#include "torch2pnnx.h"
#include <dlfcn.h>

namespace pnnx {

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

int torch2pnnx(const std::string& ptPath,
               Graph& g,
               const std::string& device,
               const std::vector<std::vector<int64_t>>& inputShapes,
               const std::vector<std::string>& inputTypes,
               const std::vector<std::vector<int64_t>>& inputShapes2,
               const std::vector<std::string>& inputTypes2,
               const std::vector<std::string>& customOpModules,
               const std::vector<std::string>& moduleOperators,
               const std::string& foldableConstantsZippath,
               std::set<std::string>& foldableConstants) {
    for (auto& m: customOpModules) {
        std::cerr << "load custom module: " << m << std::endl;
        void* handle = dlopen(m.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "dlopen " << m << " failed " << dlerror() << std::endl;
        }
    }

    std::vector<at::Tensor> inputTensors;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        const std::vector<int64_t>& shape = inputShapes[i];
        const std::string& type = inputTypes[i];
        at::Tensor t = torch::ones(shape, InputType2C10ScalarType(type));
        if (device == "gpu") {
            t = t.cuda();
        }
        inputTensors.push_back(t);
    }

    std::vector<at::Tensor> inputTensors2;
    for (size_t i = 0; i < inputShapes2.size(); ++i) {
        const std::vector<int64_t>& shape = inputShapes2[i];
        const std::string& type = inputTypes2[i];
        at::Tensor t = torch::ones(shape, InputType2C10ScalarType(type));
        if (device == "gpu") {
            t = t.cuda();
        }
        inputTensors2.push_back(t);
    }

    torch::jit::Module mod;
    try {
        mod = torch::jit::load(ptPath, (device == "gpu") ? c10::kCUDA : c10::kCPU);
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

    auto graph = OptimizeTorchScript(mod,
                                     inputTensors,
                                     inputTensors2,
                                     moduleOperators,
                                     ptPath,
                                     device,
                                     foldableConstants,
                                     foldableConstantsZippath);

    return 0;
}

}// namespace pnnx