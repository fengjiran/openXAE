//
// Created by richard on 7/3/24.
//

#include "shape_inference.h"

#include "storezip.h"

namespace pnnx {

static bool IsInplaceOp(const std::string& opType) {
    auto size = opType.size();
    return size > 2 && opType[size - 2] != '_' && opType[size - 1] == '_';
}

static bool IsAliasOp(const std::string& opType) {
    return opType == "aten::slice" || opType == "aten::select" || opType == "aten::view";
}

static bool IsStaticShapeFoldable(const std::string& opType) {
    return opType == "aten::size" ||
           opType == "aten::new_empty" ||
           opType == "aten::new_full" ||
           opType == "aten::new_ones" ||
           opType == "aten::new_zeros" ||
           opType == "aten::empty_like" ||
           opType == "aten::full_like" ||
           opType == "aten::ones_like" ||
           opType == "aten::zeros_like" ||
           opType == "aten::_shape_as_tensor";
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

static bool ValueLinkOutput(const torch::jit::Value* v, const std::vector<torch::jit::Value*>& outputs) {
    for (auto x: outputs) {
        if (v == x) {
            return true;
        }
    }

    for (auto it: v->uses()) {
        auto node = it.user;
        for (auto x: node->outputs()) {
            bool link = ValueLinkOutput(x, outputs);
            if (link) {
                return true;
            }
        }

        std::string opName = node->kind().toDisplayString();
        if (IsInplaceOp(opName)) {
            return true;
        }
    }
    return false;
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

    // bookkeep foldable tensors
    std::unordered_map<std::string, int> valueLinkInputMap;
    {
        // build value alias map for inplace op
        std::unordered_map<std::string, torch::jit::Value*> valueAliasMap;
        for (const auto& n: graph->block()->nodes()) {
            if (n->kind() == c10::prim::GetAttr ||
                n->kind() == c10::prim::Constant ||
                n->kind() == c10::prim::CallMethod) {
                continue;
            }

            std::string opType = n->kind().toDisplayString();
            if ((!IsInplaceOp(opType) && !IsAliasOp(opType)) || n->inputs().empty() || n->outputs().empty()) {
                continue;
            }

            std::string is = n->input(0)->debugName();
            if (is.empty()) {
                continue;
            }

            for (size_t i = 0; i < n->outputs().size(); ++i) {
                auto out2 = n->output(i);
                auto tensorType = out2->type()->cast<torch::jit::TensorType>();
                if (!tensorType) {
                    continue;
                }

                std::string os = out2->debugName();
                if (os.empty()) {
                    continue;
                }

                if (valueAliasMap.find(is) == valueAliasMap.end()) {
                    valueAliasMap[os] = n->input(0);
                } else {
                    valueAliasMap[os] = valueAliasMap[is];
                }
            }
        }

        for (const auto& x: valueAliasMap) {
            std::cerr << "alias " << x.first << " -> " << x.second->debugName() << std::endl;
        }

        bool ignoreAtenSize = inputTensors2.empty();
        for (size_t i = 1; i < graph->inputs().size(); ++i) {
            auto in0 = graph->inputs()[i];
            for (auto it: in0->uses()) {
                auto node = it.user;
                BuildValueLinkInputMap(node, valueAliasMap, valueLinkInputMap, ignoreAtenSize);
            }
        }

        for (const auto& x: valueLinkInputMap) {
            std::cerr << "link_input " << x.first << " " << x.second << std::endl;
        }
    }

    StoreZipWriter zip;
    zip.open(foldableConstantsZippath);
    for (size_t p = 0; p < moreValueNames.size(); ++p) {
        std::unordered_set<std::string>& valueNames = moreValueNames[p];
        std::vector<torch::jit::Value*>& values = moreValues[p];
        
    }
}

}// namespace pnnx
