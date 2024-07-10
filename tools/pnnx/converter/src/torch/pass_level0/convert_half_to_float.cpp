//
// Created by richard on 7/8/24.
//

#include "torch/pass_level0.h"

namespace pnnx {

void ConvertHalf2Float(torch::jit::Module& mod) {
    for (auto subMod: mod.children()) {
        ConvertHalf2Float(subMod);
    }

    for (auto namedAttr: mod.named_attributes(false)) {
        const std::string& name = namedAttr.name;
        auto attr = namedAttr.value;
        if (attr.type()->kind() == c10::TypeKind::TensorType) {
            auto t = attr.toTensor();
            if (t.scalar_type() == c10::ScalarType::Half || t.scalar_type() == c10::ScalarType::BFloat16) {
                at::Tensor tFP32 = t.toType(c10::ScalarType::Float);
                mod.setattr(name, tFP32);
            }
        }
    }
}

}// namespace pnnx
