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

}// namespace pnnx