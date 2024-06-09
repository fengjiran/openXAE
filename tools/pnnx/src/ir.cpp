//
// Created by 赵丹 on 24-5-30.
//
#include "ir.h"
#include "storezip.h"
#include <cfloat>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace pnnx {

static bool IsInteger(DataType type) {
    bool flag;
    switch (type) {
        case DataType::kDataTypeInt32:
        case DataType::kDataTypeInt64:
        case DataType::kDataTypeInt16:
        case DataType::kDataTypeInt8:
        case DataType::kDataTypeUInt8:
        case DataType::kDataTypeBool:
            flag = true;
            break;

        case DataType::kDataTypeUnknown:
        case DataType::kDataTypeFloat32:
        case DataType::kDataTypeFloat64:
        case DataType::kDataTypeFloat16:
        case DataType::kDataTypeComplex64:
        case DataType::kDataTypeComplex128:
        case DataType::kDataTypeComplex32:
        case DataType::kDataTypeBFloat16:
            flag = false;
            break;

        default:
            flag = false;
    }

    return flag;
}

static std::string DataType2String(DataType type) {
    std::string str;
    switch (type) {
        case DataType::kDataTypeFloat32:
            str = "f32";
            break;
        case DataType::kDataTypeFloat64:
            str = "f64";
            break;
        case DataType::kDataTypeFloat16:
            str = "f16";
            break;
        case DataType::kDataTypeInt32:
            str = "i32";
            break;
        case DataType::kDataTypeInt64:
            str = "i64";
            break;
        case DataType::kDataTypeInt16:
            str = "i16";
            break;
        case DataType::kDataTypeInt8:
            str = "i8";
            break;
        case DataType::kDataTypeUInt8:
            str = "u8";
            break;
        case DataType::kDataTypeBool:
            str = "bool";
            break;
        case DataType::kDataTypeComplex64:
            str = "c64";
            break;
        case DataType::kDataTypeComplex128:
            str = "c128";
            break;
        case DataType::kDataTypeComplex32:
            str = "c32";
            break;
        case DataType::kDataTypeBFloat16:
            str = "bf16";
            break;
        case DataType::kDataTypeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

static const char* DataType2NumpyString(DataType type) {
    const char* str;
    switch (type) {
        case DataType::kDataTypeFloat32:
            str = "float32";
            break;
        case DataType::kDataTypeFloat64:
            str = "float64";
            break;
        case DataType::kDataTypeFloat16:
            str = "float16";
            break;
        case DataType::kDataTypeInt32:
            str = "int32";
            break;
        case DataType::kDataTypeInt64:
            str = "int64";
            break;
        case DataType::kDataTypeInt16:
            str = "int16";
            break;
        case DataType::kDataTypeInt8:
            str = "int8";
            break;
        case DataType::kDataTypeUInt8:
            str = "uint8";
            break;
        case DataType::kDataTypeBool:
            str = "bool8";
            break;
        case DataType::kDataTypeComplex64:
            str = "csingle";
            break;
        case DataType::kDataTypeComplex128:
            str = "cdouble";
            break;
        case DataType::kDataTypeComplex32:
            str = "chalf";
            break;
        case DataType::kDataTypeBFloat16:
            str = "bfloat16";
            break;
        case DataType::kDataTypeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

static const char* DataType2TorchString(DataType type) {
    const char* str;
    switch (type) {
        case DataType::kDataTypeFloat32:
            str = "torch.float";
            break;
        case DataType::kDataTypeFloat64:
            str = "torch.double";
            break;
        case DataType::kDataTypeFloat16:
            str = "torch.half";
            break;
        case DataType::kDataTypeInt32:
            str = "torch.int";
            break;
        case DataType::kDataTypeInt64:
            str = "torch.long";
            break;
        case DataType::kDataTypeInt16:
            str = "torch.short";
            break;
        case DataType::kDataTypeInt8:
            str = "torch.int8";
            break;
        case DataType::kDataTypeUInt8:
            str = "torch.uint8";
            break;
        case DataType::kDataTypeBool:
            str = "torch.bool";
            break;
        case DataType::kDataTypeComplex64:
            str = "torch.complex64";
            break;
        case DataType::kDataTypeComplex128:
            str = "torch.complex128";
            break;
        case DataType::kDataTypeComplex32:
            str = "torch.complex32";
            break;
        case DataType::kDataTypeBFloat16:
            str = "torch.bfloat16";
            break;
        case DataType::kDataTypeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

static size_t SizeOf(DataType type) {
    size_t size;
    switch (type) {
        case DataType::kDataTypeUnknown:
            size = 0;
            break;

        case DataType::kDataTypeInt8:
        case DataType::kDataTypeUInt8:
        case DataType::kDataTypeBool:
            size = 1;
            break;

        case DataType::kDataTypeInt16:
        case DataType::kDataTypeFloat16:
        case DataType::kDataTypeBFloat16:
            size = 2;
            break;

        case DataType::kDataTypeInt32:
        case DataType::kDataTypeFloat32:
        case DataType::kDataTypeComplex32:
            size = 4;
            break;

        case DataType::kDataTypeInt64:
        case DataType::kDataTypeFloat64:
        case DataType::kDataTypeComplex64:
            size = 8;
            break;

        case DataType::kDataTypeComplex128:
            size = 16;
            break;
    }
    return size;
}

static DataType String2Type(const std::string& s) {
    if (s == "f32") return DataType::kDataTypeFloat32;
    if (s == "f64") return DataType::kDataTypeFloat64;
    if (s == "f16") return DataType::kDataTypeFloat16;
    if (s == "i32") return DataType::kDataTypeInt32;
    if (s == "i64") return DataType::kDataTypeInt64;
    if (s == "i16") return DataType::kDataTypeInt16;
    if (s == "i8") return DataType::kDataTypeInt8;
    if (s == "u8") return DataType::kDataTypeUInt8;
    if (s == "bool") return DataType::kDataTypeBool;
    if (s == "c64") return DataType::kDataTypeComplex64;
    if (s == "c128") return DataType::kDataTypeComplex128;
    if (s == "c32") return DataType::kDataTypeComplex32;
    if (s == "bf16") return DataType::kDataTypeBFloat16;
    return DataType::kDataTypeUnknown;
}

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

Attribute::Attribute(const std::vector<int>& shape, const std::vector<float>& t)
    : type_(DataType::kDataTypeFloat32), shape_(shape) {
    if (!shape_.empty()) {
        data_.resize(size() * GetElemSize());
        memcpy((void*) data_.data(), (const void*) t.data(), data_.size());
    }
}

size_t Attribute::GetElemSize() const {
    return SizeOf(type_);
}

size_t Attribute::size() const {
    if (shape_.empty()) {
        return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
}

std::vector<float> Attribute::CastToFloat32() const {
    std::vector<float> v(size());
    if (type() == DataType::kDataTypeFloat32) {
        memcpy((void*) v.data(), (const void*) data_.data(), data_.size());
    } else if (type() == DataType::kDataTypeFloat64) {
        const auto* p = (const double*) data_.data();
        for (auto& item: v) {
            item = static_cast<float>(*p++);
        }
    } else if (type() == DataType::kDataTypeFloat16) {
        const auto* p = (const unsigned short*) data_.data();
        for (auto& item: v) {
            item = float16_to_float32(*p++);
        }
    } else {
        std::cerr << "Cannot convert to float32 type.\n";
    }
    return v;
}

void Attribute::SetFloat32Data(const std::vector<float>& newData) {
    data_.resize(newData.size() * GetElemSize());
    switch (type()) {
        case DataType::kDataTypeFloat32: {
            memcpy((void*) data_.data(), (const void*) newData.data(), data_.size());
            break;
        }

        case DataType::kDataTypeFloat64: {
            auto* p = (double*) data_.data();
            for (const auto& item: newData) {
                *p = item;
                ++p;
            }
            break;
        }

        case DataType::kDataTypeFloat16: {
            auto* p = (unsigned short*) data_.data();
            for (const auto& item: newData) {
                *p = float32_to_float16(item);
                ++p;
            }
        }

        default:
            std::cerr << "Cannot convert to float32 type.\n";
    }
}

bool operator==(const Attribute& lhs, const Attribute& rhs) {
    if (lhs.type() != rhs.type()) {
        return false;
    }

    if (lhs.type() == DataType::kDataTypeUnknown) {
        return true;
    }

    if (lhs.GetShape() != rhs.GetShape()) {
        return false;
    }

    if (lhs.GetRawData() != rhs.GetRawData()) {
        return false;
    }
    return true;
}

Attribute operator+(const Attribute& a, const Attribute& b) {
    Attribute c;
    if (a.type() != b.type()) {
        std::cerr << "concat attribute type mismatch\n";
        return c;
    }

    if (a.GetShape() != b.GetShape()) {
        std::cerr << "concat attribute shape mismatch\n";
        return c;
    }

    c.SetType(a.type());
    c.GetShape() = a.GetShape();
    c.GetShape()[0] += b.GetShape()[0];// concat the first dim

    c.GetRawData().resize(a.GetRawData().size() + b.GetRawData().size());
    memcpy(c.GetRawData().data(), a.GetRawData().data(), a.GetRawData().size());
    memcpy(c.GetRawData().data() + a.GetRawData().size(), b.GetRawData().data(), b.GetRawData().size());
    return c;
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
//    op->GetParameters()[key] = Parameter::ParseFromString(value);
    op->GetParameters()[key] = std::make_shared<Parameter>(Parameter::ParseFromString(value));
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
            operand->GetShape().push_back(-1);
        } else if (elem[0] == '%') {
            // this shape is a variable,
            // encode %abc as symbolic tag
            operand->GetShape().push_back(-233);
            size_t index = operand->GetShape().size() - 1;
            std::string s = elem.substr(1);
            operand->GetParams()[std::string("__shape__") + std::to_string(index)] = Parameter(s);
        } else {
            operand->GetShape().push_back(std::stoi(elem));
        }
    }
}

static void LoadAttribute(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value, StoreZipReader& szr) {
    op->GetAttributes()[key] = std::make_shared<Attribute>();
    std::shared_ptr<Attribute>& attr = op->GetAttributes()[key];

    // parse attribute data type
    std::string str = value.substr(value.find_last_of(')') + 1);
    DataType type = String2Type(str);
    if (type == DataType::kDataTypeUnknown) {
        return;
    }

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
            paramFile << " " << it.first << "=" << Parameter::Encode2String(*it.second);
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
                if (x == -1) {
                    paramFile << (--size ? "?," : "?");
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
                if (x == -1) {
                    paramFile << (--size ? "?," : "?");
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


//Graph::~Graph() {
//    ops_.clear();
//    operands_.clear();
//}

}// namespace pnnx