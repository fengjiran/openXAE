//
// Created by richard on 7/10/24.
//

#include "pass_level1.h"

namespace pnnx {

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

}
