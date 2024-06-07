//
// Created by 赵丹 on 24-5-30.
//
#include "ir.h"
#include "storezip.h"
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stack>

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

static std::string DataTypeToString(DataType type) {
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

static const char* type_to_numpy_string(DataType type) {
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

static const char* type_to_dtype_string(DataType type) {
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

static DataType string_to_type(const char* s) {
    if (std::strcmp(s, "f32") == 0) return DataType::kDataTypeFloat32;
    if (std::strcmp(s, "f64") == 0) return DataType::kDataTypeFloat64;
    if (std::strcmp(s, "f16") == 0) return DataType::kDataTypeFloat16;
    if (std::strcmp(s, "i32") == 0) return DataType::kDataTypeInt32;
    if (std::strcmp(s, "i64") == 0) return DataType::kDataTypeInt64;
    if (std::strcmp(s, "i16") == 0) return DataType::kDataTypeInt16;
    if (std::strcmp(s, "i8") == 0) return DataType::kDataTypeInt8;
    if (std::strcmp(s, "u8") == 0) return DataType::kDataTypeUInt8;
    if (std::strcmp(s, "bool") == 0) return DataType::kDataTypeBool;
    if (std::strcmp(s, "c64") == 0) return DataType::kDataTypeComplex64;
    if (std::strcmp(s, "c128") == 0) return DataType::kDataTypeComplex128;
    if (std::strcmp(s, "c32") == 0) return DataType::kDataTypeComplex32;
    if (std::strcmp(s, "bf16") == 0) return DataType::kDataTypeBFloat16;
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
            fprintf(stderr, "unknown parameter type %d\n", static_cast<int>(param.type()));
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

Attribute::Attribute(const std::initializer_list<int>& shape, const std::vector<float>& t)
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
        fprintf(stderr, "cannot convert type %d to float32 data\n", static_cast<int>(type_));
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
            fprintf(stderr, "cannot convert float32 newData to type_ %d\n", static_cast<int>(type_));
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
        fprintf(stderr, "concat attribute type_ mismatch\n");
        return c;
    }

    if (a.GetShape() != b.GetShape()) {
        fprintf(stderr, "concat attribute shape mismatch\n");
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

void Operand::remove_consumer(const Operator* op) {
    auto it = std::find(consumers.begin(), consumers.end(), op);
    if (it != consumers.end()) {
        consumers.erase(it);
    }
}

Operand* Operator::named_input(const std::string& key) {
    for (size_t i = 0; i < input_names.size(); ++i) {
        if (key == input_names[i]) {
            return inputs[i];
        }
    }
    return nullptr;
}

const Operand* Operator::named_input(const std::string& key) const {
    for (size_t i = 0; i < input_names.size(); ++i) {
        if (key == input_names[i]) {
            return inputs[i];
        }
    }
    return nullptr;
}

static void load_parameter(Operator* op, const std::string& key, const std::string& value) {
    op->params[key] = Parameter::ParseFromString(value);
}

static void load_input_key(Operator* op, const std::string& key, const std::string& value) {
    op->input_names.resize(op->inputs.size());
    for (size_t i = 0; i < op->inputs.size(); ++i) {
        const auto* operand = op->inputs[i];
        if (operand->name_ == value) {
            op->input_names[i] = key;
            break;
        }
    }
}

static void load_shape(Operator* op, const std::string& key, const std::string& value) {
    Operand* operand = nullptr;
    for (const auto r: op->inputs) {
        if (r->name_ == key) {
            operand = r;
            break;
        }
    }

    if (!operand) {
        for (const auto r: op->outputs) {
            if (r->name_ == key) {
                operand = r;
                break;
            }
        }
    }

    if (!operand) {
        fprintf(stderr, "no such operand %s for operator %s\n", key.c_str(), op->name_.c_str());
        return;
    }

    // parse type, e.g. #input=(1,3,10,10)f32
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    operand->SetType(string_to_type(typestr.c_str()));

    // parse shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);
    operand->shape_.clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');
        if (elem == "?") {
            operand->shape_.push_back(-1);
        } else if (elem[0] == '%') {
            // encode %abc as symbolic tag
            operand->shape_.push_back(-233);
            size_t index = operand->shape_.size() - 1;
            std::string s = elem.substr(1);
            operand->params[std::string("__shape__") + std::to_string(index)] = Parameter(s);
        } else {
            operand->shape_.push_back(std::stoi(elem));
        }
    }
}

static void load_attribute(Operator* op, const std::string& key, const std::string& value, StoreZipReader& szr) {
    Attribute& a = op->attrs[key];

    // parse type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    a.SetType(string_to_type(typestr.c_str()));
    //    a.type_ = string_to_type(typestr.c_str());

    if (a.type() == DataType::kDataTypeUnknown) {
        return;
    }

    // parse shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);
    a.GetShape().clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');
        a.GetShape().push_back(std::stoi(elem));
    }

    if (a.GetShape().empty()) {
        return;
    }

    // parse data
    size_t size = std::accumulate(a.GetShape().begin(), a.GetShape().end(), 1, std::multiplies<>());
    size_t bytesize = size * SizeOf(a.type());

    std::string filename = op->name_ + "." + key;
    size_t filesize = szr.get_file_size(filename);
    if (filesize == 0) {
        // no such file
        return;
    }

    if (filesize != bytesize) {
        fprintf(stderr, "file size not match expect %lu but got %lu\n", bytesize, filesize);
    }

    a.GetRawData().resize(bytesize);
    szr.read_file(filename, (char*) a.GetRawData().data());
}

int Graph::save(const std::string& paramPath, const std::string& binPath) {
    FILE* paramfp = fopen(paramPath.c_str(), "wb");
    if (!paramfp) {
        fprintf(stderr, "fopen %s failed!\n", paramPath.c_str());
        return -1;
    }

    StoreZipWriter szw;
    if (szw.open(binPath) != 0) {
        fprintf(stderr, "open %s failed!\n", binPath.c_str());
        return -1;
    }

    // magic number
    fprintf(paramfp, "7767517\n");

    // op number and operand number
    fprintf(paramfp, "%d %d\n", (int) ops.size(), (int) operands.size());

    // dump op info of graph
    for (const auto* op: ops) {
        // dump basic info of op
        fprintf(paramfp, "%-24s %-24s %d %d", op->type_.c_str(), op->name_.c_str(), (int) op->inputs.size(), (int) op->outputs.size());

        // dump op input operand info
        for (const auto* operand: op->inputs) {
            fprintf(paramfp, " %s", operand->name_.c_str());
        }

        // dump op output operand info
        for (const auto* operand: op->outputs) {
            fprintf(paramfp, " %s", operand->name_.c_str());
        }

        // dump op param info
        for (const auto& it: op->params) {
            fprintf(paramfp, " %s=", it.first.c_str());
            std::string s = Parameter::Encode2String(it.second);
            fprintf(paramfp, "%s", s.c_str());
        }

        // dump op attrs info
        for (const auto& it: op->attrs) {
            fprintf(paramfp, " @%s=", it.first.c_str());
            fprintf(paramfp, "(");
            const Attribute& attr = it.second;

            size_t size = attr.GetShape().size();
            for (const auto& x: attr.GetShape()) {
                fprintf(paramfp, "%d%s", x, (--size ? "," : ""));
            }
            fprintf(paramfp, ")");
            fprintf(paramfp, "%s", DataTypeToString(attr.type()).c_str());

            std::string filename = op->name_ + "." + it.first;
            szw.write_file(filename, attr.GetRawData().data(), attr.GetRawData().size());
        }

        if (op->input_names.size() == op->inputs.size()) {
            for (size_t i = 0; i < op->inputs.size(); ++i) {
                if (op->input_names[i].empty()) {
                    continue;
                }
                const auto* operand = op->inputs[i];
                fprintf(paramfp, " $%s=%s", op->input_names[i].c_str(), operand->name_.c_str());
            }
        }

        for (const auto* operand: op->inputs) {
            if (operand->shape_.empty()) {
                continue;
            }

            fprintf(paramfp, " #%s=", operand->name_.c_str());
            fprintf(paramfp, "(");

            size_t size = operand->shape_.size();
            for (const auto& x: operand->shape_) {
                if (x == -1) {
                    fprintf(paramfp, "%s", (--size ? "?," : "?"));
                } else {
                    fprintf(paramfp, "%d%s", x, (--size ? "," : ""));
                }
            }

            fprintf(paramfp, ")");
            fprintf(paramfp, "%s", DataTypeToString(operand->type()).c_str());
        }

        for (const auto* operand: op->outputs) {
            if (operand->shape_.empty()) {
                continue;
            }

            fprintf(paramfp, " #%s=", operand->name_.c_str());
            fprintf(paramfp, "(");

            size_t size = operand->shape_.size();
            for (const auto& x: operand->shape_) {
                if (x == -1) {
                    fprintf(paramfp, "%s", (--size ? "?," : "?"));
                } else {
                    fprintf(paramfp, "%d%s", x, (--size ? "," : ""));
                }
            }

            fprintf(paramfp, ")");
            fprintf(paramfp, "%s", DataTypeToString(operand->type()).c_str());
        }
        fprintf(paramfp, "\n");
    }
    fclose(paramfp);

    return 0;
}

int Graph::load(const std::string& paramPath, const std::string& binPath) {
    std::ifstream inFile(paramPath, std::ios::in | std::ios::binary);
    if (!inFile.is_open()) {
        std::fprintf(stderr, "param file open failed.\n");
        return -1;
    }

    StoreZipReader szr;
    if (szr.open(binPath) != 0) {
        std::fprintf(stderr, "bin file open failed.\n");
        return -1;
    }

    // parse the first line, magic number
    int magicNum = 0;
    {
        std::string line;
        std::getline(inFile, line);
        std::istringstream iss(line);
        iss >> magicNum;
    }

    // parse the second line, operator number and operand number
    int operatorNum = 0;
    int operandNum = 0;
    {
        std::string line;
        std::getline(inFile, line);
        std::istringstream iss(line);
        iss >> operatorNum >> operandNum;
    }

    for (int i = 0; i < operatorNum; ++i) {
        std::string line;
        std::getline(inFile, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;

        int inputNum = 0;
        int outputNum = 0;

        iss >> type >> name >> inputNum >> outputNum;

        Operator* op = CreateOperator(type, name);
        for (int j = 0; j < inputNum; ++j) {
            std::string operand_name;
            iss >> operand_name;
            Operand* r = GetOperand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < outputNum; ++j) {
            std::string operand_name;
            iss >> operand_name;
            Operand* r = CreateOperand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
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
                // attribute
                load_attribute(op, key.substr(1), value, szr);
            } else if (key[0] == '$') {
                // operand input key
                load_input_key(op, key.substr(1), value);
            } else if (key[0] == '#') {
                // operand shape
                load_shape(op, key.substr(1), value);
            } else {
                // parameter
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

Operator* Graph::CreateOperator(const std::string& type, const std::string& name) {
    auto* op = new Operator;
    op->type_ = type;
    op->name_ = name;
    ops.push_back(op);
    return op;
}

Operator* Graph::CreateOperatorBefore(const std::string& type, const std::string& name, const Operator* cur) {
    auto* op = new Operator;
    op->type_ = type;
    op->name_ = name;
    ops.insert(std::find(ops.begin(), ops.end(), cur), op);
    return op;
}

Operator* Graph::CreateOperatorAfter(const std::string& type, const std::string& name, const Operator* cur) {
    auto* op = new Operator;
    op->type_ = type;
    op->name_ = name;
    ops.insert(std::find(ops.begin(), ops.end(), cur) + 1, op);
    return op;
}

Operand* Graph::CreateOperand(const std::string& name) {
    auto* r = new Operand;
    r->name_ = name;
    operands.push_back(r);
    return r;
}

Operand* Graph::GetOperand(const std::string& name) {
    for (auto* r: operands) {
        if (r->name_ == name) {
            return r;
        }
    }
    return nullptr;
}

const Operand* Graph::GetOperand(const std::string& name) const {
    for (const auto* r: operands) {
        if (r->name_ == name) {
            return r;
        }
    }
    return nullptr;
}

Graph::~Graph() {
    for (auto x: ops) {
        delete x;
    }

    for (auto x: operands) {
        delete x;
    }

    ops.clear();
    operands.clear();
}

}// namespace pnnx