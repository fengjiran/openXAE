//
// Created by 赵丹 on 24-5-30.
//
#include "ir.h"
#include "storezip.h"
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace pnnx {

std::string Parameter::Encode2String(const Parameter& param) {
    std::string code;
    switch (param.type()) {
        case ParameterType::kParameterUnknown: {
            code = "None";
            break;
        }

        case ParameterType::kParameterBool: {
            code = param.toBool() ? "True" : "False";
            break;
        }

        case ParameterType::kParameterInt: {
            code = std::to_string(param.toInt());
            break;
        }

        case ParameterType::kParameterFloat: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%e", param.toFloat());
            code = buf;
            break;
        }

        case ParameterType::kParameterString: {
            code = param.toString();
            break;
        }

        case ParameterType::kParameterArrayInt: {
            code += "(";
            size_t size = param.toArrayInt().size();
            for (const auto& ele: param.toArrayInt()) {
                code += (std::to_string(ele) + (--size ? "," : ""));
            }
            code += ")";
            break;
        }

        case ParameterType::kParameterArrayFloat: {
            code += "(";
            size_t size = param.toArrayFloat().size();
            for (const auto& ele: param.toArrayFloat()) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%e", ele);
                code += (std::string(buf) + (--size ? "," : ""));
            }
            code += ")";
            break;
        }

        case ParameterType::kParameterArrayString: {
            code += "(";
            size_t size = param.toArrayString().size();
            for (const auto& ele: param.toArrayString()) {
                code += (ele + (--size ? "," : ""));
            }
            code += ")";
            break;
        }

        case ParameterType::kParameterComplex: {
            char buf[128];
            snprintf(buf, sizeof(buf), "%e+%ei", param.toComplex().real(), param.toComplex().imag());
            code = buf;
            break;
        }

        case ParameterType::kParameterArrayComplex: {
            code += "(";
            size_t size = param.toArrayComplex().size();
            for (const auto& ele: param.toArrayComplex()) {
                char buf[128];
                snprintf(buf, sizeof(buf), "%e+%ei", ele.real(), ele.imag());
                code += (std::string(buf) + (--size ? "," : ""));
            }
            code += ")";
            break;
        }

        default:
            std::cerr << "Unknown parameter type.\n";
    }

    return code;
}

Parameter Parameter::ParseFromString(const std::string& value) {
    // string type
    if (value.find('%') != std::string::npos) {
        Parameter p(value);
        return p;
    }

    // null type
    if (value == "None" || value == "()" || value == "[]") {
        return {};
    }

    // bool type
    if (value == "True" || value == "False") {
        return Parameter(value == "True");
    }

    // array
    if (value[0] == '(' || value[0] == '[') {
        Parameter p;
        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);
        while (!lcss.eof()) {
            std::string elem;
            std::getline(lcss, elem, ',');
            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                // array string
                p.SetType(ParameterType::kParameterArrayString);
                p.AddElemToArrayString(elem);
            } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                // array float
                p.SetType(ParameterType::kParameterArrayFloat);
                p.AddElemToArrayFloat(std::stof(elem));
            } else {
                // array integer
                p.SetType(ParameterType::kParameterArrayInt);
                p.AddElemToArrayInt(std::stoi(elem));
            }
        }
        return p;
    }

    // string
    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
        Parameter p(value);
        return p;
    }

    // float
    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
        return Parameter(std::stof(value));
    }

    // integer
    return Parameter(std::stoi(value));
}

bool operator==(const Parameter& lhs, const Parameter& rhs) {
    if (lhs.type() != rhs.type()) {
        return false;
    }

    if (lhs.type() == ParameterType::kParameterUnknown) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterBool && lhs.toBool() == rhs.toBool()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterInt && lhs.toInt() == rhs.toInt()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterFloat && std::abs(lhs.toFloat() - rhs.toFloat()) < FLT_EPSILON) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterString && lhs.toString() == rhs.toString()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterArrayInt && lhs.toArrayInt() == rhs.toArrayInt()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterArrayFloat && lhs.toArrayFloat() == rhs.toArrayFloat()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterArrayString && lhs.toArrayString() == rhs.toArrayString()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterComplex && lhs.toComplex() == rhs.toComplex()) {
        return true;
    }

    if (lhs.type() == ParameterType::kParameterArrayComplex && lhs.toArrayComplex() == rhs.toArrayComplex()) {
        return true;
    }

    return false;
}


void Operand::RemoveConsumer(const std::shared_ptr<Operator>& op) {
    auto it = std::find(consumers_.begin(), consumers_.end(), op);
    if (it != consumers_.end()) {
        consumers_.erase(it);
    }
}

std::shared_ptr<Operand> Operator::GetNamedInput(const std::string& key) const {
    for (size_t i = 0; i < inputNames_.size(); ++i) {
        if (key == inputNames_[i]) {
            return inputOperands_[i];
        }
    }
    return {};
}


static void LoadParameter(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    op->GetParameters()[key] = std::make_shared<ParameterVar>(CreateParameterFromString(value));
}

static void LoadInputName(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    op->GetInputNames().resize(op->GetInputOperands().size());
    for (size_t i = 0; i < op->GetInputOperands().size(); ++i) {
        const auto& operand = op->GetInputOperands()[i];
        if (operand->name() == value) {
            op->GetInputNames()[i] = key;
            break;
        }
    }
}

static void LoadOperand(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    std::shared_ptr<Operand> operand;
    for (const auto& r: op->GetInputOperands()) {
        if (r->name() == key) {
            operand = r;
            break;
        }
    }

    if (!operand) {
        for (const auto& r: op->GetOutputOperands()) {
            if (r->name() == key) {
                operand = r;
                break;
            }
        }
    }

    if (!operand) {
        std::cerr << "operator " << op->name() << " has no such operand! " << key << std::endl;
        return;
    }

    // parse data type, e.g. #input=(1,3,10,10)f32
    std::string str = value.substr(value.find_last_of(')') + 1);
    operand->SetType(String2Type(str));

    // parse shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);
    operand->GetShape().clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');
        if (elem == "?") {
            operand->GetShape().push_back(DimUnknownTag);
        } else if (elem[0] == '%') {
            // this shape is a variable,
            // encode %abc as symbolic tag
            operand->GetShape().push_back(DimVariableTag);
            size_t index = operand->GetShape().size() - 1;
            std::string s = elem.substr(1);
            operand->GetParams()[std::string("__shape__") + std::to_string(index)] = std::make_shared<ParameterVar>(Parameter_(s));
        } else {
            operand->GetShape().push_back(std::stoi(elem));
        }
    }
}

static void LoadAttribute(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value, StoreZipReader& szr) {
    // parse attribute data type
    std::string str = value.substr(value.find_last_of(')') + 1);
    DataType type = String2Type(str);
    if (type == DataType::kDataTypeUnknown) {
        return;
    }

    op->GetAttributes()[key] = std::make_shared<Attribute>();
    std::shared_ptr<Attribute>& attr = op->GetAttributes()[key];

    attr->SetType(type);

    // parse shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);
    attr->GetShape().clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');
        attr->GetShape().push_back(std::stoi(elem));
    }

    if (attr->GetShape().empty()) {
        return;
    }

    // parse data
    size_t sizeInByte =
            std::accumulate(attr->GetShape().begin(), attr->GetShape().end(), 1, std::multiplies<>()) * SizeOf(type);

    std::string filename = op->name() + "." + key;
    size_t filesize = szr.get_file_size(filename);
    if (filesize == 0) {
        // no such file
        return;
    }

    if (filesize != sizeInByte) {
        std::cerr << "file size not match, expect " << sizeInByte << " but got " << filesize << std::endl;
    }

    attr->GetRawData().resize(sizeInByte);
    szr.read_file(filename, (char*) attr->GetRawData().data());
}

int Graph::save(const std::string& paramPath, const std::string& binPath) {
    std::ofstream paramFile(paramPath, std::ios::out | std::ios::binary);
    if (!paramFile.is_open()) {
        std::cerr << "param file " << paramPath << " open failed!\n";
        return -1;
    }

    StoreZipWriter szw;
    if (szw.open(binPath) != 0) {
        std::cerr << "bin file " << binPath << " open failed!\n";
        return -1;
    }

    // magic number
    paramFile << "7767517" << std::endl;

    // op number and operand number
    paramFile << static_cast<int>(ops_.size()) << " " << static_cast<int>(operands_.size()) << std::endl;

    // dump op info
    for (const auto& op: ops_) {
        paramFile << std::left << std::setw(24) << op->type() << " "
                  << std::left << std::setw(24) << op->name() << " "
                  << static_cast<int>(op->GetInputOperands().size()) << " "
                  << static_cast<int>(op->GetOutputOperands().size());

        // dump op input operand name
        for (const auto& operand: op->GetInputOperands()) {
            paramFile << " " << operand->name();
        }

        // dump op output operand name
        for (const auto& operand: op->GetOutputOperands()) {
            paramFile << " " << operand->name();
        }

        // dump op param info
        for (const auto& it: op->GetParameters()) {
            std::string value;
            auto visitor = [&value](const auto& arg) { value = arg.Encode2String(); };
            std::visit(visitor, *it.second);
            paramFile << " " << it.first << "=" << value;
        }

        // dump op attrs info
        for (const auto& it: op->GetAttributes()) {
            paramFile << " @" << it.first << "=(";
            const auto& attr = it.second;
            size_t size = attr->GetShape().size();
            for (const auto& x: attr->GetShape()) {
                paramFile << x << (--size ? "," : "");
            }
            paramFile << ")" << DataType2String(attr->type());

            std::string filename = op->name() + "." + it.first;
            szw.write_file(filename, attr->GetRawData().data(), attr->GetRawData().size());
        }

        if (op->GetInputNames().size() == op->GetInputOperands().size()) {
            for (size_t i = 0; i < op->GetInputOperands().size(); ++i) {
                if (op->GetInputNames()[i].empty()) {
                    continue;
                }
                const auto& operand = op->GetInputOperands()[i];
                paramFile << " $" << op->GetInputNames()[i] << "=" << operand->name();
            }
        }

        // dump input operands
        for (const auto& operand: op->GetInputOperands()) {
            if (operand->GetShape().empty()) {
                continue;
            }

            paramFile << " #" << operand->name() << "=(";
            size_t size = operand->GetShape().size();
            for (const auto& x: operand->GetShape()) {
                if (x == DimUnknownTag) {
                    paramFile << (--size ? "?," : "?");
                } else if (x == DimVariableTag) {
                    // %abc, shape variable
                    size_t idx = operand->GetShape().size() - size;
                    std::string key = "__shape__" + std::to_string(idx);
                    std::string value;
                    auto visitor = [&value](const auto& arg) { value = arg.Encode2String(); };
                    std::visit(visitor, *(operand->GetParams()[key]));
                    paramFile << "%" << value << (--size ? "," : "");
                } else {
                    paramFile << x << (--size ? "," : "");
                }
            }

            paramFile << ")" << DataType2String(operand->type());
        }

        // dump output operands
        for (const auto& operand: op->GetOutputOperands()) {
            if (operand->GetShape().empty()) {
                continue;
            }

            paramFile << " #" << operand->name() << "=(";
            size_t size = operand->GetShape().size();
            for (const auto& x: operand->GetShape()) {
                if (x == DimUnknownTag) {
                    paramFile << (--size ? "?," : "?");
                } else if (x == DimVariableTag) {
                    // %abc, shape variable
                    size_t idx = operand->GetShape().size() - size;
                    std::string key = "__shape__" + std::to_string(idx);
                    std::string value;
                    auto visitor = [&value](const auto& arg) { value = arg.Encode2String(); };
                    std::visit(visitor, *(operand->GetParams()[key]));
                    paramFile << "%" << value << (--size ? "," : "");
                } else {
                    paramFile << x << (--size ? "," : "");
                }
            }

            paramFile << ")" << DataType2String(operand->type());
        }
        paramFile << std::endl;
    }
    paramFile.close();
    return 0;
}

int Graph::load(const std::string& paramPath, const std::string& binPath) {
    std::ifstream paramFile(paramPath, std::ios::in | std::ios::binary);
    if (!paramFile.is_open()) {
        std::cerr << "param file " << paramPath << " open failed!\n";
        return -1;
    }

    StoreZipReader szr;
    if (szr.open(binPath) != 0) {
        std::cerr << "bin file " << binPath << " open failed!\n";
        return -1;
    }

    // parse the first line, magic number
    int magicNum = 0;
    {
        std::string line;
        std::getline(paramFile, line);
        std::istringstream iss(line);
        iss >> magicNum;
    }

    // parse the second line, operator number and operand number
    int operatorNum = 0;
    int operandNum = 0;
    {
        std::string line;
        std::getline(paramFile, line);
        std::istringstream iss(line);
        iss >> operatorNum >> operandNum;
    }

    for (int i = 0; i < operatorNum; ++i) {
        std::string line;
        std::getline(paramFile, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int inputOperandNum = 0;
        int outputOperandNum = 0;

        iss >> type >> name >> inputOperandNum >> outputOperandNum;

        const auto& op = CreateOperator(type, name);
        for (int j = 0; j < inputOperandNum; ++j) {
            std::string operandName;
            iss >> operandName;
            const auto& r = GetOperand(operandName);
            r->AddConsumer(op);
            op->AddInputOperand(r);
        }

        for (int j = 0; j < outputOperandNum; ++j) {
            std::string operandName;
            iss >> operandName;
            const auto& r = CreateOperand(operandName);
            r->SetProducer(op);
            op->AddOutputOperand(r);
        }

        while (!iss.eof()) {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);
            if (key[0] == '@') {
                // load attribute raw data, shape and data type
                LoadAttribute(op, key.substr(1), value, szr);
            } else if (key[0] == '$') {
                // operand input key
                LoadInputName(op, key.substr(1), value);
            } else if (key[0] == '#') {
                // load operand shape and data type
                LoadOperand(op, key.substr(1), value);
            } else {
                // load parameter
                LoadParameter(op, key, value);
            }
        }
    }

    return 0;
}

std::shared_ptr<Operand> Graph::CreateOperator(const std::string& type, const std::string& name,
                                               const std::map<std::string, std::shared_ptr<ParameterVar>>& params,
                                               const std::map<std::string, std::shared_ptr<Attribute>>& attrs,
                                               const std::vector<std::shared_ptr<Operand>>& inputOperands,
                                               const std::vector<std::string>& inputOperandNames,
                                               const std::string& outputName,
                                               DataType outputType,
                                               const std::vector<int>& outputShape) {
    auto op = std::make_shared<Operator>(name, type, params, attrs, inputOperands, inputOperandNames);
    ops_.push_back(op);

    for (const auto& it: inputOperands) {
        it->AddConsumer(op);
    }

    if (!outputName.empty() && !outputShape.empty()) {
        auto outOperand = std::make_shared<Operand>(outputName, outputType, outputShape);
        op->AddOutputOperand(outOperand);
        outOperand->SetProducer(op);

        operands_.push_back(outOperand);
        return outOperand;
    }

    return {};
}

std::shared_ptr<Operator> Graph::CreateOperator(const std::string& type, const std::string& name) {
    auto op = std::make_shared<Operator>(name, type);
    ops_.push_back(op);
    return op;
}

std::shared_ptr<Operator> Graph::CreateOperatorBefore(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur) {
    auto op = std::make_shared<Operator>(name, type);
    ops_.insert(std::find(ops_.begin(), ops_.end(), cur), op);
    return op;
}

std::shared_ptr<Operator> Graph::CreateOperatorAfter(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur) {
    auto op = std::make_shared<Operator>(name, type);
    ops_.insert(std::find(ops_.begin(), ops_.end(), cur) + 1, op);
    return op;
}

std::shared_ptr<Operand> Graph::CreateOperand(const std::string& name) {
    auto r = std::make_shared<Operand>(name);
    operands_.push_back(r);
    return r;
}

std::shared_ptr<Operand> Graph::GetOperand(const std::string& name) {
    for (const auto& r: operands_) {
        if (r->name() == name) {
            return r;
        }
    }
    return {};
}

}// namespace pnnx