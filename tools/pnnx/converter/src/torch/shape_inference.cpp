//
// Created by richard on 7/3/24.
//

#include "shape_inference.h"
#include "constant_unpooling.h"
#include "convert_half_to_float.h"
#include "flatten_input.h"
#include "inline_block.h"
#include "reset_device.h"
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

        torch::jit::Module mod2 = torch::jit::load(ptpath, (device == "gpu") ? c10::kCUDA : c10::kCPU);
        mod2.eval();
        ConvertHalf2Float(mod2);

        auto method = mod2.find_method("forward");
        if (!method) {
            method = mod2.get_methods()[0];
        }
        auto graph2 = method->graph();
        inline_block(graph2, moduleOperators);
        ResetDevice(graph2, device);
        FlattenInput(graph2);
        ConstantUnpooling(graph2);

        std::vector<torch::jit::Value*> values2;
        for (auto n: graph2->nodes()) {
            for (const auto& v: n->outputs()) {
                auto tensorType = v->type()->cast<torch::jit::TensorType>();
                if (!tensorType) {
                    continue;
                }

                if (valueNames.find(v->debugName()) != valueNames.end()) {
                    values2.push_back(v);
                }
            }
        }
        std::cerr << "\n----------------\n\n";

        // set new graph output
        torch::jit::Node* newRetNode = graph2->createTuple(at::ArrayRef<torch::jit::Value*>(values2));
        graph2->appendNode(newRetNode);
        graph2->eraseOutput(0);
        graph2->registerOutput(newRetNode->outputs()[0]);

        // construct schema for new inputs and outputs
        {
            auto oldfs = method->function().getSchema();
            std::vector<c10::Argument> arguments;
            std::vector<c10::Argument> returns;

            for (size_t i = 0; i < graph2->inputs().size(); ++i) {
                auto v = graph2->inputs()[i];
                arguments.emplace_back(v->debugName(), v->type());
            }

            for (size_t i = 0; i < graph2->outputs().size(); ++i) {
                auto v = graph2->outputs()[i];
                returns.emplace_back(v->debugName(), v->type());
            }

            c10::FunctionSchema newfs(oldfs.name(), oldfs.overload_name(), arguments, returns);
            method->function().setSchema(newfs);
        }

        // inference for all tensots
        auto outputs = mod2.copy().get_method(method->name())(inputs).toTuple();
        if (inputTensors2.empty()) {
            // assign shape info
            for (size_t i = 0; i < values2.size(); ++i) {
                auto v = values[i];
                auto t = outputs->elements()[i].toTensor();
                v->setType(c10::TensorType::create(t));

                // check if value that does not depend on inputs
                if (valueLinkInputMap.find(v->debugName()) == valueLinkInputMap.end() && ValueLinkOutput(v, graphOutputs)) {
                    foldableConstants.insert(v->debugName());
                    at::Tensor t2 = t.cpu().contiguous();
                    zip.write_file(v->debugName(), (const char*) t2.data_ptr(), t2.nbytes());
                }
            }
        } else {
            // assign dynamic shape info
            auto outputs2 = mod2.copy().get_method(method->name())(inputs2).toTuple();
            std::cerr << "assign dynamic shape info.\n";
            for (size_t i = 0; i < values2.size(); ++i) {
                auto v = values[i];
                auto t = outputs->elements()[i].toTensor();
                auto t2 = outputs2->elements()[i].toTensor();

                auto type1 = c10::TensorType::create(t);
                auto type2 = c10::TensorType::create(t2);

                std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
                std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

                for (size_t j = 0; j < sizes1.size(); ++j) {
                    if (sizes1[j] == sizes2[j]) {
                        continue;
                    }
                    sizes1[j] = c10::ShapeSymbol::fromStaticSize(-1);
                }

                auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));
                v->setType(finaltype);

                // check if value that does not depend on inputs
                if (valueLinkInputMap.find(v->debugName()) == valueLinkInputMap.end() && ValueLinkOutput(v, graphOutputs)) {
                    foldableConstants.insert(v->debugName());
                    at::Tensor t21 = t.cpu().contiguous();
                    zip.write_file(v->debugName(), (const char*) t21.data_ptr(), t21.nbytes());
                }
            }
        }
    }
    zip.close();
    if (inputTensors2.empty()) {
        for (size_t i = 0; i < inputTensors2.size(); ++i) {
            auto type = c10::TensorType::create(inputTensors[i]);
            graph->inputs()[i + 1]->setType(type);
        }
    } else {
        for (size_t i = 0; i < inputTensors.size(); ++i) {
            auto type1 = c10::TensorType::create(inputTensors[i]);
            auto type2 = c10::TensorType::create(inputTensors2[i]);

            std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
            std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

            for (size_t j = 0; j < sizes1.size(); ++j) {
                if (sizes1[j] == sizes2[j]) {
                    continue;
                }
                sizes1[i] = c10::ShapeSymbol::fromStaticSize(-1);
            }

            auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));
            graph->outputs()[i + 1]->setType(finaltype);
        }
    }
}

}// namespace pnnx
