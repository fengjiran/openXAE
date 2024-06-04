//
// Created by 赵丹 on 24-5-30.
//
#include "ir.h"
#include "storezip.h"
#include "utils.h"
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stack>

namespace pnnx {

static bool type_is_integer(DataType type) {
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

static const char* type_to_string(DataType type) {
    const char* str;
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

static size_t type_to_elemsize(DataType type) {
    size_t elemsize;
    switch (type) {
        case DataType::kDataTypeFloat32:
            elemsize = 4;
            break;
        case DataType::kDataTypeFloat64:
            elemsize = 8;
            break;
        case DataType::kDataTypeFloat16:
            elemsize = 2;
            break;
        case DataType::kDataTypeInt32:
            elemsize = 4;
            break;
        case DataType::kDataTypeInt64:
            elemsize = 8;
            break;
        case DataType::kDataTypeInt16:
            elemsize = 2;
            break;
        case DataType::kDataTypeInt8:
        case DataType::kDataTypeUInt8:
        case DataType::kDataTypeBool:
            elemsize = 1;
            break;
        case DataType::kDataTypeComplex64:
            elemsize = 8;
            break;
        case DataType::kDataTypeComplex128:
            elemsize = 16;
            break;
        case DataType::kDataTypeComplex32:
            elemsize = 4;
            break;
        case DataType::kDataTypeBFloat16:
            elemsize = 2;
            break;
        case DataType::kDataTypeUnknown:
            elemsize = 0;
            break;
    }
    return elemsize;
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

std::string Parameter::encode_to_string(const Parameter& param) {
    if (param.type == ParameterType::kParameterUnknown) {
        return "None";
    }

    if (param.type == ParameterType::kParameterBool) {
        if (param.b) {
            return "True";
        } else {
            return "False";
        }
    }

    if (param.type == ParameterType::kParameterInt) {
        return std::to_string(param.i);
    }

    if (param.type == ParameterType::kParameterFloat) {
        char buf[64];
        sprintf(buf, "%e", param.f);
        return buf;
    }

    if (param.type == ParameterType::kParameterString) {
        return param.s;
    }

    if (param.type == ParameterType::kParameterArrayInt) {
        std::string s("(");
        size_t size = param.ai.size();
        for (const auto& ele: param.ai) {
            s += (std::to_string(ele) + (--size ? "," : ""));
        }
        s += ")";
        return s;
    }

    if (param.type == ParameterType::kParameterArrayFloat) {
        std::string s("(");
        size_t size = param.af.size();
        for (const auto& ele: param.af) {
            char buf[64];
            sprintf(buf, "%e", ele);
            s += (std::string(buf) + (--size ? "," : ""));
        }
        s += ")";
        return s;
    }

    if (param.type == ParameterType::kParameterArrayString) {
        std::string s("(");
        size_t size = param.as.size();
        for (const auto& ele: param.as) {
            s += (ele + (--size ? "," : ""));
        }
        s += ")";
        return s;
    }

    if (param.type == ParameterType::kParameterComplex) {
        char buf[128];
        sprintf(buf, "%e+%ei", param.c.real(), param.c.imag());
        return buf;
    }

    if (param.type == ParameterType::kParameterArrayComplex) {
        std::string s("(");
        size_t size = param.ac.size();
        for (const auto& ele: param.ac) {
            char buf[128];
            sprintf(buf, "%e+%ei", ele.real(), ele.imag());
            s += (std::string(buf) + (--size ? "," : ""));
        }
        s += ")";
        return s;
    }

    fprintf(stderr, "unknown parameter type %d\n", static_cast<int>(param.type));
    return {};
}

Parameter Parameter::parse_from_string(const std::string& value) {
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
                p.type = ParameterType::kParameterArrayString;
                p.as.push_back(elem);
            } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                // array float
                p.type = ParameterType::kParameterArrayFloat;
                p.af.push_back(std::stof(elem));
            } else {
                // array integer
                p.type = ParameterType::kParameterArrayInt;
                p.ai.push_back(std::stoi(elem));
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
    if (lhs.type != rhs.type) {
        return false;
    }

    if (lhs.type == ParameterType::kParameterUnknown) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterBool && lhs.b == rhs.b) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterInt && lhs.i == rhs.i) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterFloat && std::abs(lhs.f - rhs.f) < FLT_EPSILON) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterString && lhs.s == rhs.s) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterArrayInt && lhs.ai == rhs.ai) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterArrayFloat && lhs.af == rhs.af) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterArrayString && lhs.as == rhs.as) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterComplex && lhs.c == rhs.c) {
        return true;
    }

    if (lhs.type == ParameterType::kParameterArrayComplex && lhs.ac == rhs.ac) {
        return true;
    }

    return false;
}

Attribute::Attribute(const std::initializer_list<int>& shape_, const std::vector<float>& t)
    : type(DataType::kDataTypeFloat32), shape(shape_) {
    if (!shape.empty()) {
        data.resize(elemcount() * elemsize());
        memcpy((void*) data.data(), (const void*) t.data(), data.size());
    }
}

size_t Attribute::elemsize() const {
    return type_to_elemsize(type);
}

int Attribute::elemcount() const {
    if (shape.empty()) {
        return 0;
    }

    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

std::vector<float> Attribute::get_float32_data() const {
    std::vector<float> v(elemcount());
    if (type == DataType::kDataTypeFloat32) {
        memcpy((void*) v.data(), (const void*) data.data(), data.size());
    } else if (type == DataType::kDataTypeFloat64) {
        const auto* p = (const double*) data.data();
        for (auto& item: v) {
            item = static_cast<float>(*p++);
        }
    } else if (type == DataType::kDataTypeFloat16) {
        const auto* p = (const unsigned short*) data.data();
        for (auto& item: v) {
            item = float16_to_float32(*p++);
        }
    } else {
        fprintf(stderr, "cannot convert type %d to float32 data\n", static_cast<int>(type));
    }
    return v;
}

void Attribute::set_float32_data(const std::vector<float>& data_) {
    data.resize(data_.size() * elemsize());
    switch (type) {
        case DataType::kDataTypeFloat32: {
            memcpy((void*) data.data(), (const void*) data_.data(), data.size());
            break;
        }

        case DataType::kDataTypeFloat64: {
            auto* p = (double*) data.data();
            for (const auto& item: data_) {
                *p = item;
                ++p;
            }
            break;
        }

        case DataType::kDataTypeFloat16: {
            auto* p = (unsigned short*) data.data();
            for (const auto& item: data_) {
                *p = float32_to_float16(item);
                ++p;
            }
        }

        default:
            fprintf(stderr, "cannot convert float32 data to type %d\n", static_cast<int>(type));
    }
}

bool operator==(const Attribute& lhs, const Attribute& rhs) {
    if (lhs.type != rhs.type) {
        return false;
    }

    if (lhs.type == DataType::kDataTypeUnknown) {
        return true;
    }

    if (lhs.shape != rhs.shape) {
        return false;
    }

    if (lhs.data != rhs.data) {
        return false;
    }
    return true;
}

Attribute operator+(const Attribute& a, const Attribute& b) {
    Attribute c;
    if (a.type != b.type) {
        fprintf(stderr, "concat attribute type mismatch\n");
        return c;
    }

    if (a.shape != b.shape) {
        fprintf(stderr, "concat attribute shape mismatch\n");
        return c;
    }

    c.type = a.type;
    c.shape = a.shape;
    c.shape[0] += b.shape[0];// concat the first dim

    c.data.resize(a.data.size() + b.data.size());
    memcpy(c.data.data(), a.data.data(), a.data.size());
    memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());
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

}// namespace pnnx