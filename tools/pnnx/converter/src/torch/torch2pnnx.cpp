//
// Created by richard on 6/22/24.
//

#include "torch2pnnx.h"

namespace pnnx {

int torch2pnnx(const std::string& ptPath,
               const std::string& device) {
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

    auto graph = OptimizeTorchScript(mod);
    graph->dump();

    return 0;
}

}// namespace pnnx