//
// Created by richard on 8/6/24.
//

#include "pnnx/converter/include/torch/pass_level2.h"

namespace pnnx {

void GraphRewriterPass::Write(const std::shared_ptr<Operator>& op,
                              const std::map<std::string, std::shared_ptr<Parameter>>& capturedParams) const {
    if (ReplacePatternGraph().empty()) {
        for (const auto& x: capturedParams) {
            op->GetParameters()[x.first] = x.second;
        }
        return;
    }

    for (const auto& x: op->GetParameters()) {
        if (x.second->type() != ParameterType::kParameterString) {
            continue;
        }

        std::string str = x.second->toString();
        if (str.find('%') == std::string::npos) {
            continue;
        }

        // search % token and replace with captured
        size_t pos = str.find('%');
        while (pos != std::string::npos) {
            // %xyz
            char buf[256];
            sscanf(str.c_str() + pos + 1, "%255[^][,() ]", buf);
            std::string key(buf);
            if (capturedParams.find(key) == capturedParams.end()) {
                std::cerr << "replace pattern param " << key << " missing captured.\n";
                return;
            }

            // replace %xyz with encoded_str
            auto encodedStr = capturedParams.at(key)->toString();
            str.replace(pos, key.size() + 1, encodedStr);
            pos = str.find('%', pos + 1);
        }

        op->GetParameters()[x.first] = std::make_shared<Parameter>(
                Parameter::CreateParameterFromString(str));
    }

    for (const auto& operand: op->GetInputOperands()) {
        auto shape = operand->GetShape();
        for (size_t i = 0; i < shape.size(); ++i) {
            int ai = shape[i];
            if (ai == DimVariableTag) {
                auto key = operand->GetParams().at(std::string("__shape__") + std::to_string(i))->toValue<std::string>();
                if (capturedParams.find(key) == capturedParams.end()) {
                    std::cerr << "replace pattern param " << key << " missing captured.\n";
                    return;
                }

                shape[i] = capturedParams.at(key)->toValue<int>();
            }
        }
    }

    for (const auto& operand: op->GetOutputOperands()) {
        auto shape = operand->GetShape();
        for (size_t i = 0; i < shape.size(); ++i) {
            int ai = shape[i];
            if (ai == DimVariableTag) {
                auto key = operand->GetParams().at(std::string("__shape__") + std::to_string(i))->toValue<std::string>();
                if (capturedParams.find(key) == capturedParams.end()) {
                    std::cerr << "replace pattern param " << key << " missing captured.\n";
                    return;
                }

                shape[i] = capturedParams.at(key)->toValue<int>();
            }
        }
    }
}

void GraphRewriterPass::Write(const std::shared_ptr<Operator>& op,
                              const std::map<std::string, std::shared_ptr<Parameter>>& capturedParams,
                              const std::map<std::string, std::shared_ptr<Attribute>>& capturedAttrs) const {
    Write(op, capturedParams);
    for (auto& x: op->GetAttributes()) {
        if (x.second->type() != DataType::kDataTypeUnknown) {
            continue;
        }

        std::string key(x.second->GetRawData().data());
        if (key.empty()) {
            continue;
        }

        //        op->GetAttributes()[x.first] = std::make_shared<Attribute>(capturedAttrs.at(key));
        x.second = capturedAttrs.at(key);
    }
}

void GraphRewriterPass::Write(const std::map<std::string, std::shared_ptr<Operator>>& ops,
                              const std::map<std::string, std::shared_ptr<Parameter>>& capturedParams) const {
    for (const auto& x: ops) {
        Write(x.second, capturedParams);
    }
}

void GraphRewriterPass::Write(const std::map<std::string, std::shared_ptr<Operator>>& ops,
                              const std::map<std::string, std::shared_ptr<Parameter>>& capturedParams,
                              const std::map<std::string, std::shared_ptr<Attribute>>& capturedAttrs) const {
    Write(ops, capturedParams);
    for (const auto& x: ops) {
        for (auto& attr: x.second->GetAttributes()) {
            if (attr.second->type() != DataType::kDataTypeUnknown) {
                continue;
            }

            std::string key(attr.second->GetRawData().data());
            if (key.empty() || key[0] != '%') {
                continue;
            }

            attr.second = capturedAttrs.at(key.substr(1));
        }
    }
}

static bool TokenIsArgument(const std::string& t) {
    if (t[0] != '@' || t.size() < 2) {
        return false;
    }

    for (size_t i = 1; i < t.size(); i++) {
        if (t[i] < '0' || t[i] > '9') {
            return false;
        }
    }

    return true;
}

static bool MatchExpression(const std::shared_ptr<Operator>& op1,
                            const std::shared_ptr<Operator>& op2,
                            std::map<std::string, std::shared_ptr<Parameter>>& capturedParams) {
    if (op1->GetParameters().size() != 1 ||
        op1->GetParameters().find("expr") == op1->GetParameters().end()) {
        return false;
    }

    if (op2->GetParameters().size() != 1 ||
        op2->GetParameters().find("expr") == op2->GetParameters().end()) {
        return false;
    }

    const auto& expr1 = op1->GetParameters().at("expr")->toValue<std::string>();
    const auto& expr2 = op2->GetParameters().at("expr")->toValue<std::string>();
    if (expr1 == expr2) {
        return true;
    }

    // split expr1 into tokens
    std::vector<std::string> tokens1;
    std::vector<std::string> tokens2;
    {
        std::string t;
        for (char ch: expr1) {
            if (ch == '[') {// list
                t += ch;
                tokens1.push_back(t);
                t.clear();
            } else if (ch == ']') {
                if (!t.empty()) {
                    tokens1.push_back(t);
                    t.clear();
                }
                t += ch;
            } else if (ch == '(' || ch == ')' || ch == ',') {
                if (!t.empty()) {
                    tokens1.push_back(t);
                    t.clear();
                }
            } else {
                t += ch;
            }
        }
        if (!t.empty()) {
            tokens1.push_back(t);
        }
    }

    // split expr2 into tokens
    {
        std::string t;
        for (char ch: expr2) {
            if (ch == '[') {// list
                t += ch;
                tokens2.push_back(t);
                t.clear();
            } else if (ch == ']') {
                if (!t.empty()) {
                    tokens2.push_back(t);
                    t.clear();
                }
                t += ch;
            } else if (ch == '(' || ch == ')' || ch == ',') {
                if (!t.empty()) {
                    tokens2.push_back(t);
                    t.clear();
                }
            } else {
                t += ch;
            }
        }
        if (!t.empty()) {
            tokens2.push_back(t);
        }
    }

    if (tokens1.size() != tokens2.size()) {
        return false;
    }

    // capture values
    for (size_t i = 0; i < tokens1.size(); ++i) {
        const auto& at = tokens1[i];
        const auto& bt = tokens2[i];

        if (at == bt) {
            continue;
        }

        if (bt[0] != '%') {
            return false;
        }

        if (TokenIsArgument(at)) {
            return false;
        }

        std::string key = bt.substr(1);
        capturedParams[key] = std::make_shared<Parameter>(Parameter::CreateParameterFromString(at));
    }

    return true;
}

static bool MatchParameter(const Parameter& a,
                           const Parameter& b,
                           std::map<std::string, std::shared_ptr<Parameter>>& capturedParams) {
    if (b.type() == ParameterType::kParameterString && b.toString()[0] == '%') {
        auto key = b.toString().substr(1);
        if (capturedParams.find(key) != capturedParams.end()) {
            // match previous captured parameter
            return *capturedParams.at(key) == a;
        }

        // captured parameter
        capturedParams[key] = std::make_shared<Parameter>(a);
        return true;
    }

    if (b.type() == ParameterType::kParameterString && b.toString() == "*") {
        // ignored parameter
        return true;
    }

    if (b.type() == ParameterType::kParameterString &&
        (b.toString()[0] == '(' || b.toString()[0] == '[') &&
        (b.toString().find('%') != std::string::npos)) {
        // list with pattern
        if (a.type() != ParameterType::kParameterArrayInt &&
            a.type() != ParameterType::kParameterArrayFloat &&
            a.type() != ParameterType::kParameterArrayString) {
            return false;
        }

        std::string lc = b.toString().substr(1, b.toString().size() - 2);
        std::istringstream lcss(lc);

        size_t i = 0;
        while (!lcss.eof()) {
            std::string elem;
            std::getline(lcss, elem, ',');

            if (elem[0] == '%') {
                std::string key = elem.substr(1);
                if (capturedParams.find(key) != capturedParams.end()) {
                    // match previous captured parameter
                    if (a.type() == ParameterType::kParameterArrayInt &&
                        capturedParams.at(key)->toValue<int>() != a.toValue<std::vector<int>>()[i]) {
                        return false;
                    }

                    if (a.type() == ParameterType::kParameterArrayFloat &&
                        capturedParams.at(key)->toValue<float>() != a.toValue<std::vector<float>>()[i]) {
                        return false;
                    }

                    if (a.type() == ParameterType::kParameterArrayString &&
                        capturedParams.at(key)->toValue<std::string>() != a.toValue<std::vector<std::string>>()[i]) {
                        return false;
                    }
                }

                // captured parameter
                if (a.type() == ParameterType::kParameterArrayInt) {
                    capturedParams[key] = std::make_shared<Parameter>(a.toValue<std::vector<int>>()[i]);
                }

                if (a.type() == ParameterType::kParameterArrayFloat) {
                    capturedParams[key] = std::make_shared<Parameter>(a.toValue<std::vector<float>>()[i]);
                }

                if (a.type() == ParameterType::kParameterArrayString) {
                    capturedParams[key] = std::make_shared<Parameter>(a.toValue<std::vector<std::string>>()[i]);
                }
            } else if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) ||
                       (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                // string
                if (a.type() != ParameterType::kParameterArrayString) {
                    return false;
                }

                if (a.toValue<std::vector<std::string>>()[i] != elem) {
                    return false;
                }
            } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                // float
                if (a.type() != ParameterType::kParameterArrayFloat) {
                    return false;
                }

                if (a.toValue<std::vector<float>>()[i] != std::stof(elem)) {
                    return false;
                }
            } else {
                // integer
                if (a.type() != ParameterType::kParameterArrayInt) {
                    return false;
                }

                if (a.toValue<std::vector<int>>()[i] != std::stoi(elem)) {
                    return false;
                }
            }

            i++;
        }
        return true;
    }

    if (a.type() != b.type()) {
        if (a.type() == ParameterType::kParameterInt && b.type() == ParameterType::kParameterFloat) {
            return a.toValue<int>() == b.toValue<float>();
        }

        if (a.type() == ParameterType::kParameterFloat && b.type() == ParameterType::kParameterInt) {
            return a.toValue<float>() == b.toValue<int>();
        }

        return false;
    }

    return a == b;
}

static bool MatchAttribute(const Attribute& a,
                           const Attribute& b,
                           std::map<std::string, std::shared_ptr<Parameter>>& capturedParams,
                           const std::string& attrName,
                           std::map<std::string, std::shared_ptr<Attribute>>& capturedAttrs) {
    // @data
    // @data=(1,2,3,4)f32
    // @data=%op1.data
    if (b.type() == DataType::kDataTypeUnknown) {
        std::string bs(b.GetRawData().data());
        if (bs.empty()) {
            // capture any shape
            capturedAttrs[attrName] = std::make_shared<Attribute>(a);
            return true;
        }

        if (bs[0] == '%') {
            // the captured replace
            return true;
        }

        std::cerr << "malformed attribute pattern " << bs << std::endl;
        return false;
    }

    const auto& a_shape = a.GetShape();
    const auto& b_shape = b.GetShape();

    if (b_shape.empty()) {
        return false;
    }

    if (a_shape.empty()) {
        return false;
    }

    if (a_shape.size() != b_shape.size()) {
        return false;
    }

    for (size_t i = 0; i < a_shape.size(); i++) {
        int ai = a_shape[i];
        int bi = b_shape[i];
        if (ai == bi) {
            continue;
        }

        if (bi == DimUnknownTag) {
            continue;
        }

        if (bi > 0) {
            return false;
        }

        if (bi != DimVariableTag) {
            return false;
        }

        std::string key = b.GetParameters().at(std::string("__shape__") + std::to_string(i))->toValue<std::string>();
        if (capturedParams.find(key) != capturedParams.end()) {
            // match previous captured parameter
            if (capturedParams.at(key)->toValue<int>() != ai) {
                return false;
            }
        }

        // captured parameter
        capturedParams[key] = std::make_shared<Parameter>(ai);
    }

    capturedAttrs[attrName] = std::make_shared<Attribute>(a);
    return true;
}

static bool MatchOperator(const std::shared_ptr<Operator>& a,
                          const std::shared_ptr<Operator>& b,
                          std::map<std::string, std::shared_ptr<Parameter>>& capturedParams,
                          std::map<std::string, std::shared_ptr<Attribute>>& capturedAttrs) {
    if (!(a->type() == b->type() &&
          a->GetInputOperands().size() == b->GetInputOperands().size() &&
          a->GetOutputOperands().size() == b->GetOutputOperands().size())) {
        return false;
    }

    //    if (a->type() != b->type()) {
    //        return false;
    //    }
    //
    //    if (a->GetInputOperands().size() != b->GetInputOperands().size()) {
    //        return false;
    //    }
    //
    //    if (a->GetOutputOperands().size() != b->GetOutputOperands().size()) {
    //        return false;
    //    }

    // match params
    if (b->GetParameters().size() == 1 &&
        b->GetParameters().find("%*") != b->GetParameters().end() &&
        b->GetParameters().at("%*")->type() == ParameterType::kParameterString &&
        b->GetParameters().at("%*")->toValue<std::string>() == "%*") {
        for (const auto& p: a->GetParameters()) {
            const auto& key = p.first;
            const auto& value = p.second;

            // capture all parameters
            capturedParams[b->name() + '.' + key] = value;
        }
    } else if (a->type() == "pnnx.Expression") {
        if (!MatchExpression(a, b, capturedParams))
            return false;
    } else {
        if (a->GetParameters().size() != b->GetParameters().size()) {
            return false;
        }

        for (const auto& p: a->GetParameters()) {
            const auto& akey = p.first;
            const auto& ap = p.second;

            if (b->GetParameters().find(akey) == b->GetParameters().end()) {
                return false;
            }

            if (!MatchParameter(*ap, *b->GetParameters().at(akey), capturedParams)) {
                return false;
            }
        }
    }

    // match shapes
    for (size_t i = 0; i < a->GetInputOperands().size(); ++i) {
        auto type1 = a->GetInputOperands()[i]->type();
        auto type2 = b->GetInputOperands()[i]->type();
        if (type2 != DataType::kDataTypeUnknown && type1 != type2) {
            return false;
        }

        const auto& shape1 = a->GetInputOperands()[i]->GetShape();
        const auto& shape2 = b->GetInputOperands()[i]->GetShape();
        if (shape2.empty()) {
            continue;
        }

        if (shape1.empty()) {
            return false;
        }

        if (shape1.size() != shape2.size()) {
            return false;
        }

        for (size_t j = 0; j < shape1.size(); ++j) {
            int ai = shape1[j];
            int bi = shape2[j];
            if (ai == bi) {
                continue;
            }

            if (bi == DimUnknownTag) {
                continue;
            }

            if (bi > 0) {
                return false;
            }

            if (bi != DimVariableTag) {
                return false;
            }

            std::string key = b->GetInputOperands()[i]->GetParams().at(std::string("__shape__") + std::to_string(j))->toValue<std::string>();
            if (capturedParams.find(key) != capturedParams.end()) {
                // match previous captured parameter
                if (capturedParams.at(key)->toValue<int>() != ai) {
                    return false;
                }
            }
            // captured parameter
            capturedParams[key] = std::make_shared<Parameter>(ai);
        }
    }

    for (size_t i = 0; i < a->GetOutputOperands().size(); ++i) {
        auto type1 = a->GetOutputOperands()[i]->type();
        auto type2 = b->GetOutputOperands()[i]->type();
        if (type2 != DataType::kDataTypeUnknown && type1 != type2) {
            return false;
        }

        const auto& shape1 = a->GetOutputOperands()[i]->GetShape();
        const auto& shape2 = b->GetOutputOperands()[i]->GetShape();
        if (shape2.empty()) {
            continue;
        }

        if (shape1.empty()) {
            return false;
        }

        if (shape1.size() != shape2.size()) {
            return false;
        }

        for (size_t j = 0; j < shape1.size(); ++j) {
            int ai = shape1[j];
            int bi = shape2[j];
            if (ai == bi) {
                continue;
            }

            if (bi == DimUnknownTag) {
                continue;
            }

            if (bi > 0) {
                return false;
            }

            if (bi != DimVariableTag) {
                return false;
            }

            std::string key = b->GetOutputOperands()[i]->GetParams().at(std::string("__shape__") + std::to_string(j))->toValue<std::string>();
            if (capturedParams.find(key) != capturedParams.end()) {
                // match previous captured parameter
                if (capturedParams.at(key)->toValue<int>() != ai) {
                    return false;
                }
            }
            // captured parameter
            capturedParams[key] = std::make_shared<Parameter>(ai);
        }
    }

    for (const auto& p: a->GetAttributes()) {
        const auto& akey = p.first;
        const auto& aa = p.second;
        std::string attrName = b->name() + '.' + akey;
        if (b->GetAttributes().find(akey) == b->GetAttributes().end()) {
            // capture all attributes
            capturedAttrs[attrName] = aa;
        } else {
            if (!MatchAttribute(*aa, *b->GetAttributes().at(akey), capturedParams, attrName, capturedAttrs)) {
                return false;
            }
        }
    }
    return true;
}

static bool Match(const std::shared_ptr<Operator>& anchor,
                  const std::shared_ptr<Operator>& pattern,
                  std::map<std::string, std::shared_ptr<Operator>>& matchedOperators,
                  std::map<std::string, std::shared_ptr<Operand>>& matchedInputs,
                  std::map<std::string, std::shared_ptr<Operand>>& matchedOutputs,
                  std::map<std::string, std::shared_ptr<Parameter>>& capturedParams,
                  std::map<std::string, std::shared_ptr<Attribute>>& capturedAttrs) {
    if (!MatchOperator(anchor, pattern, capturedParams, capturedAttrs)) {
        return false;
    }

    for (size_t i = 0; i < pattern->GetOutputOperands().size(); ++i) {
        const auto& consumers = pattern->GetOutputOperands()[i]->GetConsumers();
        if (consumers.size() == 1 && consumers[0]->type() == "pnnx.Output") {
            if (matchedOutputs.find(pattern->GetOutputOperands()[i]->name()) == matchedOutputs.end()) {
                matchedOutputs[pattern->GetOutputOperands()[i]->name()] = anchor->GetOutputOperands()[i];
            } else if (matchedOutputs[pattern->GetOutputOperands()[i]->name()] != anchor->GetOutputOperands()[i]) {
                return false;
            }
            continue;
        }

        if (anchor->GetOutputOperands()[i]->GetConsumers().size() != pattern->GetOutputOperands()[i]->GetConsumers().size()) {
            return false;
        }
    }

    matchedOperators[pattern->name()] = anchor;

    for (size_t i = 0; i < anchor->GetInputOperands().size(); ++i) {
        const auto& anchor2 = anchor->GetInputOperands()[i]->GetProducer();
        const auto& pattern2 = pattern->GetInputOperands()[i]->GetProducer();
        if (pattern2->type() == "pnnx.Input") {
            if (matchedInputs.find(pattern->GetInputOperands()[i]->name()) == matchedInputs.end()) {
                matchedInputs[pattern->GetInputOperands()[i]->name()] = anchor->GetInputOperands()[i];
            } else if (matchedInputs[pattern->GetInputOperands()[i]->name()] != anchor->GetInputOperands()[i]) {
                return false;
            }
            continue;
        }
        if (!Match(anchor2, pattern2, matchedOperators, matchedInputs,
                   matchedOutputs, capturedParams, capturedAttrs)) {
            return false;
        }
    }
    return true;
}

void PNNXGraphRewrite(Graph& graph,
                      const std::shared_ptr<GraphRewriterPass>& pass,
                      int& opIdx) {
    Graph patternGraph;
    patternGraph.parse(pass->MatchPatternGraph());

    // collect pattern inputs and outputs order
    std::vector<std::string> patternGraphInputs; // input operand name
    std::vector<std::string> patternGraphOutputs;// output operand name
    std::vector<std::shared_ptr<Operator>> patternGraphOutputOps;

    for (const auto& x: patternGraph.GetOperators()) {
        if (x->type() == "pnnx.Input") {
            for (const auto& y: x->GetOutputOperands()) {
                patternGraphInputs.push_back(y->name());
            }
        }

        if (x->type() == "pnnx.Output") {
            patternGraphOutputOps.push_back(x);
            for (const auto& y: x->GetInputOperands()) {
                patternGraphOutputs.push_back(y->name());
            }
        }
    }

    std::vector<std::shared_ptr<Operator>> newOps;
    while (true) {
        const int graphOpNum = (int) graph.GetOperators().size();

        bool matched = true;

        // match from output
        std::map<std::string, std::shared_ptr<Operator>> matchedOperators;
        std::map<std::string, std::shared_ptr<Operand>> matchedInputs;
        std::map<std::string, std::shared_ptr<Operand>> matchedOutputs;
        std::map<std::string, std::shared_ptr<Parameter>> capturedParams;
        std::map<std::string, std::shared_ptr<Attribute>> capturedAttrs;

        // pattern match from end to begin
        int q = graphOpNum - 1;
        for (; q >= 1; q--) {
            matched = true;
            for (const auto& pattern: patternGraphOutputOps) {
                for (const auto& operand: pattern->GetInputOperands()) {
                    const auto& pattern2 = operand->GetProducer();
                    int j = q;
                    for (; j >= 0; j--) {
                        const auto& anchor = graph.GetOperators()[j];

                        std::map<std::string, std::shared_ptr<Operator>> matchedOperators2;
                        std::map<std::string, std::shared_ptr<Operand>> matchedInputs2;
                        std::map<std::string, std::shared_ptr<Operand>> matchedOutputs2;
                        std::map<std::string, std::shared_ptr<Parameter>> capturedParams2;
                        std::map<std::string, std::shared_ptr<Attribute>> capturedAttrs2;

                        if (!Match(anchor, pattern2, matchedOperators2, matchedInputs2, matchedOutputs2,
                                   capturedParams2, capturedAttrs2)) {
                            continue;
                        }

                        bool submatch_matched = true;
                        for (const auto& x: matchedOperators2) {
                            // check these matched operators are same with previous matched ones
                            if (matchedOperators.find(x.first) != matchedOperators.end()) {
                                if (matchedOperators[x.first] != x.second) {
                                    // unmatched two sub-matches
                                    submatch_matched = false;
                                    break;
                                }
                            } else {
                                matchedOperators[x.first] = x.second;
                            }
                        }
                        if (!submatch_matched) {
                            continue;
                        }

                        for (const auto& x: matchedInputs2) {
                            if (matchedInputs.find(x.first) == matchedInputs.end()) {
                                matchedInputs[x.first] = x.second;
                            }
                        }

                        for (const auto& x: matchedOutputs2) {
                            if (matchedOutputs.find(x.first) == matchedOutputs.end()) {
                                matchedOutputs[x.first] = x.second;
                            }
                        }

                        for (const auto& x: capturedParams2) {
                            capturedParams[x.first] = x.second;
                        }

                        for (const auto& x: capturedAttrs2) {
                            capturedAttrs[x.first] = x.second;
                        }

                        // match !
                        break;
                    }

                    if (j == -1) {
                        matched = false;
                        break;
                    }
                }

                if (!matched) {
                    break;
                }
            }

            if (matched && !pass->Match(matchedOperators, capturedParams, capturedAttrs)) {
                matchedOperators.clear();
                matchedInputs.clear();
                matchedOutputs.clear();
                capturedParams.clear();
                capturedAttrs.clear();
                matched = false;
                continue;
            }

            break;
        }

        if (!matched) {
            break;
        }

        std::cerr << "matched !\n";

        // replace
        // remove all operands inside matched graph
        std::map<std::string, std::shared_ptr<Operand>> operands_to_remove;
        for (auto& _x: matchedOperators) {
            auto& x = _x.second;
            for (auto& r: x->GetInputOperands()) {
                r->RemoveConsumer(x);

                bool is_input = false;
                for (auto& r2: matchedInputs) {
                    if (r2.second == r) {
                        is_input = true;
                        break;
                    }
                }

                if (!is_input) {
                    operands_to_remove[r->name()] = r;
                }
            }
            x->GetInputOperands().clear();

            for (auto& r: x->GetOutputOperands()) {
                r->SetProducer(nullptr);

                bool is_output = false;
                for (auto& r2: matchedOutputs) {
                    if (r2.second == r) {
                        is_output = true;
                        break;
                    }
                }

                if (!is_output)
                    operands_to_remove[r->name()] = r;
            }

            x->GetOutputOperands().clear();
        }

        for (auto& _x: operands_to_remove) {
            auto& x = _x.second;
            graph.GetOperands().erase(
                    std::find(graph.GetOperands().begin(), graph.GetOperands().end(), x));
        }

        // insert new operator at the last matched one
        std::shared_ptr<Operator> cur;
        {
            size_t cur_index = 1;
            for (const auto& o: matchedOperators) {
                size_t c_index = std::find(graph.GetOperators().begin(),
                                           graph.GetOperators().end(),
                                           o.second) -
                                 graph.GetOperators().begin();
                cur_index = std::max(cur_index, c_index + 1);
            }
            cur_index = std::min(cur_index, graph.GetOperators().size() - 1);
            cur = graph.GetOperators()[cur_index];
        }

        // remove all matched operators
        for (const auto& _x: matchedOperators) {
            auto& x = _x.second;
            graph.GetOperators().erase(
                    std::find(graph.GetOperators().begin(), graph.GetOperators().end(), x));
        }

        if (pass->ReplacePatternGraph().empty()) {
            // insert single
            auto op = graph.CreateOperatorBefore(pass->TypeStr(), std::string(pass->NameStr()), cur);
            for (const auto& k: patternGraphInputs) {
                auto& r = matchedInputs.at(k);
                r->AddConsumer(op);
                op->AddInputOperand(r);
                op->GetInputNames().push_back(k);
            }

            for (const auto& k: patternGraphOutputs) {
                auto& r = matchedOutputs.at(k);
                r->SetProducer(op);
                op->AddOutputOperand(r);
            }

            pass->Write(op, capturedParams, capturedAttrs);
            newOps.push_back(op);
        } else {
            // insert multiple
            Graph replace_graph;
            replace_graph.parse(pass->ReplacePatternGraph());

            // move operators and operands from replace_graph to graph except input and output
            std::map<std::string, std::shared_ptr<Operator>> ops;

            for (auto& op: replace_graph.GetOperators()) {
                if (op->type() == "pnnx.Input" || op->type() == "pnnx.Output") {
                    continue;
                }
                graph.GetOperators().insert(std::find(graph.GetOperators().begin(), graph.GetOperators().end(), cur), op);
                ops[op->name()] = op;
                op.reset();
            }

            for (auto& r: replace_graph.GetOperands()) {
                if (r->GetProducer()->type() == "pnnx.Input" ||
                    (r->GetConsumers().size() == 1 && r->GetConsumers()[0]->type() == "pnnx.Output")) {
                    continue;
                }
                graph.GetOperands().push_back(r);
                r.reset();
            }

            replace_graph.GetOperators().erase(std::remove(replace_graph.GetOperators().begin(),
                                                           replace_graph.GetOperators().end(),
                                                           nullptr),
                                               replace_graph.GetOperators().end());
            replace_graph.GetOperands().erase(std::remove(replace_graph.GetOperands().begin(),
                                                          replace_graph.GetOperands().end(),
                                                          nullptr),
                                              replace_graph.GetOperands().end());

            for (const auto& k: patternGraphInputs) {
                auto& r = matchedInputs.at(k);
                const auto& rr = replace_graph.GetOperand(k);
                for (auto& x: rr->GetConsumers()) {
                    r->AddConsumer(x);
                    x->GetInputNames().resize(x->GetInputOperands().size());
                    for (size_t j = 0; j < x->GetInputOperands().size(); ++j) {
                        if (x->GetInputOperands()[j]->name() == k) {
                            x->GetInputOperands()[j] = r;
                            x->GetInputNames()[j] = k;
                            break;
                        }
                    }
                }
            }

            for (const auto& k: patternGraphOutputs) {
                auto& r = matchedOutputs.at(k);
                const auto& rr = replace_graph.GetOperand(k);
                r->SetProducer(rr->GetProducer());

                for (auto& operand: r->GetProducer()->GetOutputOperands()) {
                    if (operand->name() == k) {
                        operand = r;
                        break;
                    }
                }
            }

            pass->Write(ops, capturedParams, capturedAttrs);
            for (const auto& x: ops) {
                newOps.push_back(x.second);
            }
        }
    }

    // assign new op name number
    for (int i = (int) newOps.size() - 1; i >= 0; --i) {
        newOps[i]->name() = newOps[i]->name() + "_" + std::to_string(opIdx++);
    }
}

static bool IsAliasOp(const std::shared_ptr<Operator>& op) {
    if (op->type() == "aten::slice" ||
        op->type() == "aten::select" ||
        op->type() == "aten::view") {
        return true;
    }

    return false;
}

// step 1. create shadow view/slice/select/... for each consumer
// step 2. replace inplace op, append copy
// step 3. tag operand alias for view/slice/select/... output
// step 4. scan inplace op, collect affacted alias
// step 5. look for any op after the inplace op with alias input
// step 6. collect ops on the chain back to alias
// step 7. move chain after copy op
// step 8. update all alias uses after copy op, retag alias
// step 9. clear all alias tag
static void functionize(Graph& graph) {
    auto& ops = graph.GetOperators();
    const auto& operands = graph.GetOperands();

    // step1: create shadow view/slice/select for each consumer.
    {
        for (int i = (int) ops.size() - 1; i >= 0; --i) {
            const auto& op = ops[i];
            if (!IsAliasOp(op)) {
                continue;
            }

            auto& out0 = op->GetOutputOperands()[0];
            if (out0->GetConsumers().size() == 1) {
                continue;
            }

            bool allConsumersAreSame = true;
            for (size_t j = 1; j < out0->GetConsumers().size(); ++j) {
                if (out0->GetConsumers()[j] != out0->GetConsumers()[0]) {
                    allConsumersAreSame = false;
                    break;
                }
            }
            if (allConsumersAreSame) {
                continue;
            }

            for (int j = (int) out0->GetConsumers().size() - 1; j > 0; --j) {
                auto& op1 = out0->GetConsumers()[j];
                auto shadowOp = graph.CreateOperatorAfter(op->type(),
                                                          op->name() + "_pnnxshadow_" + std::to_string(j),
                                                          op);
                auto shadowOut = graph.CreateOperand(shadowOp->name() + "_out");

                shadowOp->GetInputOperands() = op->GetInputOperands();
                shadowOp->GetParameters() = op->GetParameters();
                shadowOp->AddOutputOperand(shadowOut);

                for (const auto& x: op->GetInputOperands()) {
                    x->AddConsumer(shadowOp);
                }

                shadowOut->SetProducer(shadowOp);
                shadowOut->SetType(out0->type());
                shadowOut->GetShape() = out0->GetShape();
                shadowOut->GetParams() = out0->GetParams();
                shadowOut->AddConsumer(op1);

                for (auto& operand: op1->GetInputOperands()) {
                    if (operand == out0) {
                        operand = shadowOut;
                    }
                }
            }

            out0->GetConsumers().resize(1);
        }
    }

    // step2: replace inplace op, append copy
    // step3: tag operand alias for view/slice/select/... output
    {
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            bool isInplaceOp = op->type().size() > 2 &&
                               op->type()[op->type().size() - 2] != '_' &&
                               op->type()[op->type().size() - 1] == '_';
            if (!(op->type() == "aten::copy_" || IsAliasOp(op) || isInplaceOp)) {
                continue;
            }

            const auto& in = op->GetInputOperands()[0];// tensor
            int aliasIdx;
            if (in->GetParams().find("__alias__") != in->GetParams().end()) {
                aliasIdx = in->GetParams().at("__alias__")->toValue<int>();
            } else {
                aliasIdx = std::find(operands.begin(), operands.end(), in) - operands.begin();
            }

            if (op->type() == "aten::copy_") {
                op->GetOutputOperands()[0]->GetParams()["__alias__"] = std::make_shared<Parameter>(aliasIdx);
                std::cerr << "operand " << op->GetOutputOperands()[0]->name()
                          << " is alias of " << operands[aliasIdx]->name()
                          << std::endl;

                // set copy output shape as the alias one
                op->GetOutputOperands()[0]->SetType(operands[aliasIdx]->type());
                op->GetOutputOperands()[0]->GetShape() = operands[aliasIdx]->GetShape();
                continue;
            }

            if (IsAliasOp(op)) {
                op->GetOutputOperands()[0]->GetParams()["__alias__"] = std::make_shared<Parameter>(aliasIdx);
                std::cerr << "operand " << op->GetOutputOperands()[0]->name()
                          << " is alias of " << operands[aliasIdx]->name()
                          << std::endl;
                continue;
            }

            if (isInplaceOp) {
                // replace with non-inplace version, create copy op
                op->type() = op->type().substr(0, op->type().size() - 1);

                // append aten::copy_
                if (operands[aliasIdx]->GetConsumers().size() > 1) {
                    const auto& in0 = op->GetInputOperands()[0];
                    const auto& out0 = op->GetOutputOperands()[0];

                    auto copyOp = graph.CreateOperatorAfter("aten::copy_", op->name() + "_copy", op);
                    auto copyOut = graph.CreateOperand(op->name() + "_copy_out");

                    copyOp->AddInputOperand(in0);
                    copyOp->AddInputOperand(out0);
                    in0->AddConsumer(copyOp);
                    out0->AddConsumer(copyOp);

                    copyOp->AddOutputOperand(copyOut);
                    copyOut->SetProducer(copyOp);
                }
            }
        }
    }

    // step4: scan inplace copy op, collect affected alias
    {
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            if (op->type() != "aten::copy_") {
                continue;
            }

            op->type() = "aten::copy";
            const auto& out0 = op->GetOutputOperands()[0];

            // inplace op output always alias with the input
            const int aliasIdx = out0->GetParams().at("__alias__")->toValue<int>();
            const auto& aliasIn0 = operands[aliasIdx];
            std::cerr << "---> " << op->name() << " for " << aliasIn0->name() << std::endl;

            size_t iAdvanced = 0;

            // step5: look for any op after the inplace op with alias input
            for (size_t j = i + 1; j < ops.size(); ++j) {
                auto& op1 = ops[j];
                bool affected = false;
                for (const auto& x: op1->GetInputOperands()) {
                    if (x == aliasIn0) {
                        affected = true;
                        break;
                    }

                    if (x->GetParams().find("__alias__") == x->GetParams().end()) {
                        continue;
                    }

                    int aliasIdx1 = x->GetParams().at("__alias__")->toValue<int>();
                    if (aliasIdx1 == aliasIdx) {
                        affected = true;
                        break;
                    }
                }

                if (!affected) {
                    continue;
                }

                // step6: collect ops on the chain back to alias
                std::set<size_t> chainsxOpIndexes;
                {
                    size_t op1Idx = std::find(ops.begin(), ops.end(), op1) - ops.begin();
                    if (op1Idx < i - iAdvanced) {
                        chainsxOpIndexes.insert(op1Idx);
                        std::cerr << "affected op " << op1->name() << " for "
                                  << operands[aliasIdx]->name() << std::endl;
                    }

                    while (true) {
                        const auto& x = op1->GetInputOperands()[0];
                        if (x->GetParams().find("__alias__") == x->GetParams().end()) {
                            break;
                        }

                        int aliasIdx1 = x->GetParams().at("__alias__")->toValue<int>();
                        if (aliasIdx1 != aliasIdx) {
                            break;
                        }

                        op1 = x->GetProducer();
                        size_t newOp1Idx = std::find(ops.begin(), ops.end(), op1) - ops.begin();
                        if (newOp1Idx < i - iAdvanced) {
                            chainsxOpIndexes.insert(newOp1Idx);
                            std::cerr << "affected op " << op1->name() << " for "
                                      << operands[aliasIdx]->name() << std::endl;
                        }
                    }
                }

                // step7: move chain after copy op
                {
                    int k = 0;
                    for (size_t doi: chainsxOpIndexes) {
                        doi -= k;
                        std::cerr << "---> move " << ops[doi]->name()
                                  << "after " << ops[i - iAdvanced] << std::endl;

                        for (size_t l = doi; l < i - iAdvanced; ++l) {
                            std::swap(ops[l], ops[l + 1]);
                        }

                        k++;
                    }
                    iAdvanced += chainsxOpIndexes.size();
                }

                // step8: update all alias uses after copy op, retag alias
                out0->GetParams().erase("__alias__");
                const int newAliasIdx = std::find(operands.begin(), operands.end(), out0) - operands.begin();
                for (size_t k = i - iAdvanced + 1; k < ops.size(); ++k) {
                    const auto& op2 = ops[k];

                    for (size_t l = 0; l < op2->GetInputOperands().size(); ++l) {
                        if (op2->GetInputOperands()[l] == aliasIn0) {
                            std::cerr << "---> replace " << op2->name() << "input " << op2->GetInputOperands()[l]->name()
                                      << "to " << out0->name() << std::endl;

                            op2->GetInputOperands()[l] = out0;
                            aliasIn0->RemoveConsumer(op2);
                            out0->AddConsumer(op2);
                        }
                    }

                    for (const auto& x: op2->GetOutputOperands()) {
                        if (x->GetParams().find("__alias__") != x->GetParams().end() &&
                            x->GetParams().at("__alias__")->toValue<int>() == aliasIdx) {
                            x->GetParams()["__alias__"] = std::make_shared<Parameter>(newAliasIdx);
                        }
                    }
                }

                // rewind to the updated copy op
                j -= chainsxOpIndexes.size();
            }
        }
    }

    // step9: clear all alias tag
    {
        for (const auto& x: operands) {
            x->GetParams().erase("__alias__");
        }
    }
}

void pass_level2(Graph& pg) {
    functionize(pg);
    int opIdx = 0;
    for (const auto& x: GraphRewriterPassRegistry::GetInstance().GetGlobalPNNXGraphRewriterPass()) {
        for (const auto& rewriter: x.second) {
            PNNXGraphRewrite(pg, rewriter, opIdx);
        }
    }
}

}// namespace pnnx
