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






}// namespace pnnx