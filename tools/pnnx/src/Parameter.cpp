//
// Created by 赵丹 on 24-8-17.
//

#include "Parameter.h"

namespace pnnx {

bool operator==(const Parameter& lhs, const Parameter& rhs) {
    if (lhs.type() != rhs.type()) {
        return false;
    }

    bool isEqual;
    switch (lhs.type()) {
        case ParameterType::kParameterUnknown: {
            isEqual = true;
            break;
        }
        case ParameterType::kParameterBool: {
            isEqual = lhs.toValue<bool>() == rhs.toValue<bool>();
            break;
        }
        case ParameterType::kParameterInt: {
            isEqual = lhs.toValue<int>() == rhs.toValue<int>();
            break;
        }
        case ParameterType::kParameterFloat: {
            isEqual = std::abs(lhs.toValue<float>() - rhs.toValue<float>()) <=
                      std::numeric_limits<float>::epsilon();
            break;
        }

        case ParameterType::kParameterString:{
            isEqual = lhs.toValue<std::string>() == rhs.toValue<std::string>();
            break;
        }

        case ParameterType::kParameterArrayInt:{
            isEqual = lhs.toValue<std::vector<int>>() == rhs.toValue<std::vector<int>>();
            break;
        }

        case ParameterType::kParameterArrayFloat:{
            const auto& a = lhs.toValue<std::vector<float>>();
            const auto& b = rhs.toValue<std::vector<float>>();
            isEqual = std::equal(a.begin(), a.end(), b.begin(), b.end());
            break;
        }

        case ParameterType::kParameterArrayString: {
            const auto& a = lhs.toValue<std::vector<std::string>>();
            const auto& b = rhs.toValue<std::vector<std::string>>();
            isEqual = std::equal(a.begin(), a.end(), b.begin(), b.end());
            break;
        }

        case ParameterType::kParameterComplex: {
            isEqual = lhs.toValue<std::complex<float>>() == rhs.toValue<std::complex<float>>();
            break;
        }

        case ParameterType::kParameterArrayComplex: {
            const auto& a = lhs.toValue<std::vector<std::complex<float>>>();
            const auto& b = rhs.toValue<std::vector<std::complex<float>>>();
            isEqual = std::equal(a.begin(), a.end(), b.begin(), b.end());
            break;
        }

        default:{
            isEqual = false;
            break;
        }
    }

    return isEqual;
}

Parameter Parameter::CreateParameterFromString(const std::string& value) {
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

}// namespace pnnx
