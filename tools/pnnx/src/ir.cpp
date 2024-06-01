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
    if (strcmp(s, "f32") == 0) return AttributeType::kAttributeFloat32;
    if (strcmp(s, "f64") == 0) return AttributeType::kAttributeFloat64;
    if (strcmp(s, "f16") == 0) return AttributeType::kAttributeFloat16;
    if (strcmp(s, "i32") == 0) return AttributeType::kAttributeInt32;
    if (strcmp(s, "i64") == 0) return AttributeType::kAttributeInt64;
    if (strcmp(s, "i16") == 0) return AttributeType::kAttributeInt16;
    if (strcmp(s, "i8") == 0) return AttributeType::kAttributeInt8;
    if (strcmp(s, "u8") == 0) return AttributeType::kAttributeUInt8;
    if (strcmp(s, "bool") == 0) return AttributeType::kAttributeBool;
    if (strcmp(s, "c64") == 0) return AttributeType::kAttributeComplex64;
    if (strcmp(s, "c128") == 0) return AttributeType::kAttributeComplex128;
    if (strcmp(s, "c32") == 0) return AttributeType::kAttributeComplex32;
    if (strcmp(s, "bf16") == 0) return AttributeType::kAttributeBFloat16;
    return AttributeType::kAttributeUnknown;
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

    if (lhs.type == ParameterType::kParameterFloat && lhs.f == rhs.f) {
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