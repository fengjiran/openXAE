//
// Created by 赵丹 on 24-5-30.
//
#include "ir.h"
#include "storezip.h"
#include "utils.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stack>
#include <string>
#include <cfloat>

namespace pnnx {

static bool type_is_integer(AttributeType type) {
    bool flag;
    switch (type) {
        case AttributeType::kAttributeInt32:
        case AttributeType::kAttributeInt64:
        case AttributeType::kAttributeInt16:
        case AttributeType::kAttributeInt8:
        case AttributeType::kAttributeUInt8:
        case AttributeType::kAttributeBool:
            flag = true;
            break;

        case AttributeType::kAttributeUnknown:
        case AttributeType::kAttributeFloat32:
        case AttributeType::kAttributeFloat64:
        case AttributeType::kAttributeFloat16:
        case AttributeType::kAttributeComplex64:
        case AttributeType::kAttributeComplex128:
        case AttributeType::kAttributeComplex32:
        case AttributeType::kAttributeBFloat16:
            flag = false;
            break;

        default:
            flag = false;
    }

    return flag;
}

static const char* type_to_string(AttributeType type) {
    const char* str;
    switch (type) {
        case AttributeType::kAttributeFloat32:
            str = "f32";
            break;
        case AttributeType::kAttributeFloat64:
            str = "f64";
            break;
        case AttributeType::kAttributeFloat16:
            str = "f16";
            break;
        case AttributeType::kAttributeInt32:
            str = "i32";
            break;
        case AttributeType::kAttributeInt64:
            str = "i64";
            break;
        case AttributeType::kAttributeInt16:
            str = "i16";
            break;
        case AttributeType::kAttributeInt8:
            str = "i8";
            break;
        case AttributeType::kAttributeUInt8:
            str = "u8";
            break;
        case AttributeType::kAttributeBool:
            str = "bool";
            break;
        case AttributeType::kAttributeComplex64:
            str = "c64";
            break;
        case AttributeType::kAttributeComplex128:
            str = "c128";
            break;
        case AttributeType::kAttributeComplex32:
            str = "c32";
            break;
        case AttributeType::kAttributeBFloat16:
            str = "bf16";
            break;
        case AttributeType::kAttributeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

static const char* type_to_numpy_string(AttributeType type) {
    const char* str;
    switch (type) {
        case AttributeType::kAttributeFloat32:
            str = "float32";
            break;
        case AttributeType::kAttributeFloat64:
            str = "float64";
            break;
        case AttributeType::kAttributeFloat16:
            str = "float16";
            break;
        case AttributeType::kAttributeInt32:
            str = "int32";
            break;
        case AttributeType::kAttributeInt64:
            str = "int64";
            break;
        case AttributeType::kAttributeInt16:
            str = "int16";
            break;
        case AttributeType::kAttributeInt8:
            str = "int8";
            break;
        case AttributeType::kAttributeUInt8:
            str = "uint8";
            break;
        case AttributeType::kAttributeBool:
            str = "bool8";
            break;
        case AttributeType::kAttributeComplex64:
            str = "csingle";
            break;
        case AttributeType::kAttributeComplex128:
            str = "cdouble";
            break;
        case AttributeType::kAttributeComplex32:
            str = "chalf";
            break;
        case AttributeType::kAttributeBFloat16:
            str = "bfloat16";
            break;
        case AttributeType::kAttributeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

static const char* type_to_dtype_string(AttributeType type) {
    const char* str;
    switch (type) {
        case AttributeType::kAttributeFloat32:
            str = "torch.float";
            break;
        case AttributeType::kAttributeFloat64:
            str = "torch.double";
            break;
        case AttributeType::kAttributeFloat16:
            str = "torch.half";
            break;
        case AttributeType::kAttributeInt32:
            str = "torch.int";
            break;
        case AttributeType::kAttributeInt64:
            str = "torch.long";
            break;
        case AttributeType::kAttributeInt16:
            str = "torch.short";
            break;
        case AttributeType::kAttributeInt8:
            str = "torch.int8";
            break;
        case AttributeType::kAttributeUInt8:
            str = "torch.uint8";
            break;
        case AttributeType::kAttributeBool:
            str = "torch.bool";
            break;
        case AttributeType::kAttributeComplex64:
            str = "torch.complex64";
            break;
        case AttributeType::kAttributeComplex128:
            str = "torch.complex128";
            break;
        case AttributeType::kAttributeComplex32:
            str = "torch.complex32";
            break;
        case AttributeType::kAttributeBFloat16:
            str = "torch.bfloat16";
            break;
        case AttributeType::kAttributeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

static size_t type_to_elemsize(AttributeType type) {
    size_t elemsize;
    switch (type) {
        case AttributeType::kAttributeFloat32:
            elemsize = 4;
            break;
        case AttributeType::kAttributeFloat64:
            elemsize = 8;
            break;
        case AttributeType::kAttributeFloat16:
            elemsize = 2;
            break;
        case AttributeType::kAttributeInt32:
            elemsize = 4;
            break;
        case AttributeType::kAttributeInt64:
            elemsize = 8;
            break;
        case AttributeType::kAttributeInt16:
            elemsize = 2;
            break;
        case AttributeType::kAttributeInt8:
        case AttributeType::kAttributeUInt8:
        case AttributeType::kAttributeBool:
            elemsize = 1;
            break;
        case AttributeType::kAttributeComplex64:
            elemsize = 8;
            break;
        case AttributeType::kAttributeComplex128:
            elemsize = 16;
            break;
        case AttributeType::kAttributeComplex32:
            elemsize = 4;
            break;
        case AttributeType::kAttributeBFloat16:
            elemsize = 2;
            break;
        case AttributeType::kAttributeUnknown:
            elemsize = 0;
            break;
    }
    return elemsize;
}

static AttributeType string_to_type(const char* s) {
    if (std::strcmp(s, "f32") == 0) return AttributeType::kAttributeFloat32;
    if (std::strcmp(s, "f64") == 0) return AttributeType::kAttributeFloat64;
    if (std::strcmp(s, "f16") == 0) return AttributeType::kAttributeFloat16;
    if (std::strcmp(s, "i32") == 0) return AttributeType::kAttributeInt32;
    if (std::strcmp(s, "i64") == 0) return AttributeType::kAttributeInt64;
    if (std::strcmp(s, "i16") == 0) return AttributeType::kAttributeInt16;
    if (std::strcmp(s, "i8") == 0) return AttributeType::kAttributeInt8;
    if (std::strcmp(s, "u8") == 0) return AttributeType::kAttributeUInt8;
    if (std::strcmp(s, "bool") == 0) return AttributeType::kAttributeBool;
    if (std::strcmp(s, "c64") == 0) return AttributeType::kAttributeComplex64;
    if (std::strcmp(s, "c128") == 0) return AttributeType::kAttributeComplex128;
    if (std::strcmp(s, "c32") == 0) return AttributeType::kAttributeComplex32;
    if (std::strcmp(s, "bf16") == 0) return AttributeType::kAttributeBFloat16;
    return AttributeType::kAttributeUnknown;
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

}// namespace pnnx