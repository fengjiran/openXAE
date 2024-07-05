//
// Created by richard on 7/3/24.
//

#include "shape_inference.h"

#include "storezip.h"

namespace pnnx {

static bool IsInplaceOp(const std::string& opName) {
    auto size = opName.size();
    return size > 2 && opName[size - 2] != '_' && opName[size - 1] == '_';
}

static bool IsAliasOp(const std::string& opName) {
    return opName == "aten::slice" || opName == "aten::select" || opName == "aten::view";
}

static bool IsStaticShapeFoldable(const std::string& opName) {
    return opName == "aten::size" ||
           opName == "aten::new_empty" ||
           opName == "aten::new_full" ||
           opName == "aten::new_ones" ||
           opName == "aten::new_zeros" ||
           opName == "aten::empty_like" ||
           opName == "aten::full_like" ||
           opName == "aten::ones_like" ||
           opName == "aten::zeros_like" ||
           opName == "aten::_shape_as_tensor";
}

static void BuildValueLinkInputMap(const torch::jit::Node* node,
                                   const std::unordered_map<std::string, torch::jit::Value*>& valueAliasMap,
                                   std::unordered_map<std::string, int>& valueLinkInputMap,
                                   bool ignoreAtenSize) {
    std::string opName = node->kind().toDisplayString();
    if (ignoreAtenSize && IsStaticShapeFoldable(opName)) {
        return;
    }

    for (size_t i = 0; i < node->outputs().size(); ++i) {
        auto out2 = node->outputs()[i];
        std::string os = out2->debugName();
        if (!os.empty() && valueLinkInputMap.find(os) != valueLinkInputMap.end()) {
            continue;
        }
        auto tensorType = out2->type()->cast<torch::jit::TensorType>();
        if (tensorType) {
            valueLinkInputMap[os] = 1;
        }

        for (auto it: out2->uses()) {
            auto node2 = it.user;
            BuildValueLinkInputMap(node2, valueAliasMap, valueLinkInputMap, true);
        }
    }

    if (IsInplaceOp(opName) || IsAliasOp(opName)) {
        // infect input0 and its alias
        while (true) {
            auto in2 = node->inputs()[0];
            std::string is = in2->debugName();
            if (is.empty()) {
                break;
            }
            if (valueAliasMap.find(is) == valueAliasMap.end()) {
                break;
            }

            auto in3 = valueAliasMap.at(is);
            auto tensorType = in3->type()->cast<torch::jit::TensorType>();
            if (!tensorType) {
                break;
            }

            is = in3->debugName();
            if (valueLinkInputMap.find(is) != valueLinkInputMap.end()) {
                break;
            }

            for (auto it: in3->uses()) {
                auto node2 = it.user;
                BuildValueLinkInputMap(node2, valueAliasMap, valueLinkInputMap, true);
            }
            break;
        }
    }
}

void ShapeInference(const torch::jit::Module& mod,
                    std::shared_ptr<torch::jit::Graph>& graph,
                    const std::vector<at::Tensor>& inputTensors,
                    const std::vector<at::Tensor>& inputTensors2,
                    const std::vector<std::string>& moduleOperators,
                    const std::string& ptpath,
                    const std::string& device,
                    std::set<std::string>& foldableConstants,
                    const std::string& foldableConstantsZippath) {
    // collect all intermediate output tensors
    std::vector<std::unordered_set<std::string>> moreValueNames;
    std::vector<std::vector<torch::jit::Value*>> moreValues;
    {
        std::unordered_set<std::string> valueNames;
        std::vector<torch::jit::Value*> values;
        for (const auto& n: graph->nodes()) {
            for (const auto& v: n->outputs()) {
                auto tensorType = v->type()->cast<torch::jit::TensorType>();
                if (!tensorType) {
                    continue;
                }
                valueNames.insert(v->debugName());
                values.push_back(v);
            }

            // too many intermediate blobs in one inference results in oom
            if (valueNames.size() >= 1000) {
                moreValueNames.push_back(valueNames);
                valueNames.clear();

                moreValues.push_back(values);
                values.clear();
            }
        }

        if (!valueNames.empty()) {
            moreValueNames.push_back(valueNames);
            moreValues.push_back(values);
        }
    }

    // collect graph inputs
    std::vector<torch::jit::Value*> graphInputs;
    for (size_t i = 1; i < graph->inputs().size(); ++i) {
        graphInputs.push_back(graph->inputs()[i]);
    }

    // collect graph outputs
    std::vector<torch::jit::Value*> graphOutputs;
    for (const auto& it: graph->outputs()) {
        graphOutputs.push_back(it);
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(inputTensors.size());
    for (const auto& it: inputTensors) {
        inputs.emplace_back(it);
    }

    std::vector<torch::jit::IValue> inputs2;
    inputs2.reserve(inputTensors2.size());
    for (const auto& it: inputTensors2) {
        inputs2.emplace_back(it);
    }
}

}// namespace pnnx
