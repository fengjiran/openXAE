//
// Created by richard on 6/14/24.
//

#include "Graph.h"
#include "storezip.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace pnnx {

static std::string SanitizeIdentifier(const std::string& s) {
    std::string ss = s;
    for (char& c: ss) {
        if (c == '.' || c == ':' || c == '/') {
            c = '_';
        }
    }
    return ss;
}

static Parameter CreateParameterFromString(const std::string& value) {
    // string type
    if (value.find('%') != std::string::npos) {
        return Parameter(value);
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
        bool isArrayInt = false;
        bool isArrayFloat = false;
        bool isArrayString = false;

        std::vector<int> pArrayInt;
        std::vector<float> pArrayFloat;
        std::vector<std::string> pArrayString;

        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);
        while (!lcss.eof()) {
            std::string elem;
            std::getline(lcss, elem, ',');
            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                // array string
                isArrayString = true;
                pArrayString.push_back(elem);
            } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                // array float
                isArrayFloat = true;
                pArrayFloat.push_back(std::stof(elem));
            } else {
                // array integer
                isArrayInt = true;
                pArrayInt.push_back(std::stoi(elem));
            }
        }

        if (isArrayInt && !isArrayFloat && !isArrayString) {
            return Parameter(pArrayInt);
        }

        if (!isArrayInt && isArrayFloat && !isArrayString) {
            return Parameter(pArrayFloat);
        }

        if (!isArrayInt && !isArrayFloat && isArrayString) {
            return Parameter(pArrayString);
        }

        return {};
    }

    // string
    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
        return Parameter(value);
    }

    // float
    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
        return Parameter(std::stof(value));
    }

    // integer
    return Parameter(std::stoi(value));
}

static void LoadParameter(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    op->GetParameters()[key] = std::make_shared<Parameter>(CreateParameterFromString(value));
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
            operand->GetParams()[std::string("__shape__") + std::to_string(index)] = std::make_shared<Parameter>(s);
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
            paramFile << " " << it.first << "=" << it.second->toString();
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
                    paramFile << "%" << operand->GetParams()[key]->toString() << (--size ? "," : "");
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
                    paramFile << "%" << operand->GetParams()[key]->toString() << (--size ? "," : "");
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

int Graph::python(const std::string& pyPath, const std::string& binPath) {
    std::ofstream pyfp(pyPath, std::ios::out | std::ios::binary);
    if (!pyfp.is_open()) {
        std::cerr << "python file " << pyPath << " open failed!\n";
        return -1;
    }

    pyfp << "import os\n";
    pyfp << "import numpy as np\n";
    pyfp << "import tempfile, zipfile\n";
    pyfp << "import torch\n";
    pyfp << "import torch.nn as nn\n";
    pyfp << "import torch.nn.functional as F\n";
    pyfp << "try:\n";
    pyfp << "    import torchvision\n";
    pyfp << "except:\n";
    pyfp << "    pass\n\n";

    pyfp << "class Model(nn.Module):\n";
    pyfp << "    def __init__(self):\n";
    pyfp << "        super(Model, self).__init__()\n\n";

    // module
    {
        for (const auto& op: GetOperators()) {
            if (op->type().substr(0, 3) != "nn." &&
                op->type().substr(0, 16) != "torchvision.ops.")
                continue;
            pyfp << "        self." << SanitizeIdentifier(op->name()) << " = " << op->type() << "(";
            size_t paramCnt = op->GetParameters().size();
            if (op->type() == "nn.quantized.Conv2d" || op->type() == "nn.quantized.Linear") {
                paramCnt -= 2;// ignore scale and zero_point
            }

            int paramIdx = 0;
            for (const auto& it: op->GetParameters()) {
                if (op->type() == "nn.quantized.Conv2d" || op->type() == "nn.quantized.Linear") {
                    if (it.first == "scale" || it.first == "zero_point") {
                        continue;
                    }
                }

                pyfp << it.first << "=";
                const auto& param = it.second;
                if (param->type() == ParameterType::kParameterUnknown) {
                    pyfp << "None";
                }

                if (param->type() == ParameterType::kParameterBool) {
                    if (param->toValue<bool>()) {
                        pyfp << "True";
                    } else {
                        pyfp << "False";
                    }
                }

                if (param->type() == ParameterType::kParameterInt) {
                    pyfp << param->toValue<int>();
                }

                if (param->type() == ParameterType::kParameterFloat) {
                    pyfp << param->toValue<float>();
                }

                if (param->type() == ParameterType::kParameterString) {
                    if (param->toValue<std::string>().substr(0, 6) == "torch.") {
                        pyfp << param->toValue<std::string>();
                    } else {
                        pyfp << "\'" << param->toValue<std::string>() << "\'";
                    }
                }

                if (param->type() == ParameterType::kParameterArrayInt) {
                    pyfp << "(";
                    const size_t size = param->toValue<std::vector<int>>().size();
                    for (size_t i = 0; i < size; ++i) {
                        const auto& elem = param->toValue<std::vector<int>>()[i];
                        if ((op->type() == "nn.AdaptiveAvgPool2d" ||
                             op->type() == "nn.AdaptiveAvgPool3d" ||
                             op->type() == "nn.AdaptiveMaxPool2d" ||
                             op->type() == "nn.AdaptiveMaxPool3d") &&
                            it.first == "output_size" && elem == 0) {
                            pyfp << "None";
                        } else {
                            pyfp << elem;
                        }

                        if (i + 1 != size || size == 1) {
                            pyfp << ",";
                        }
                    }
                    pyfp << ")";
                }

                if (param->type() == ParameterType::kParameterArrayFloat) {
                    pyfp << "(";
                    const size_t size = param->toValue<std::vector<float>>().size();
                    for (size_t i = 0; i < size; ++i) {
                        const auto& elem = param->toValue<std::vector<float>>()[i];
                        pyfp << elem;
                        if (i + 1 != size || size == 1) {
                            pyfp << ",";
                        }
                    }
                    pyfp << ")";
                }

                if (param->type() == ParameterType::kParameterArrayString) {
                    pyfp << "(";
                    const size_t size = param->toValue<std::string>().size();
                    for (size_t i = 0; i < size; ++i) {
                        const auto& elem = param->toValue<std::vector<std::string>>()[i];
                        if (elem.substr(0, 6) == "torch.") {
                            pyfp << elem;
                        } else {
                            pyfp << "\'" << elem << "\'";
                        }
                        if (i + 1 != size || size == 1) {
                            pyfp << ",";
                        }
                    }
                    pyfp << ")";
                }
                paramIdx++;
                if (paramIdx != paramCnt) {
                    pyfp << ", ";
                }
            }
            pyfp << ")\n";
        }
    }

    pyfp << "\n";

    // load weight
    {
        pyfp << "        archive = zipfile.ZipFile(" << "\'" << binPath << "\'" << ", \'r\')\n";
        for (const auto& op: GetOperators()) {
            if (op->type().substr(0, 3) != "nn." &&
                op->type().substr(0, 16) != "torchvision.ops.")
                continue;

        }
    }

    pyfp << "\n";

    pyfp.close();
    return 0;
}

std::shared_ptr<Operand> Graph::CreateOperator(const std::string& type, const std::string& name,
                                               const std::map<std::string, std::shared_ptr<Parameter>>& params,
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