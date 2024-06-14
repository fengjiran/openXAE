//
// Created by 赵丹 on 24-5-30.
//

#ifndef OPENXAE_IR_H
#define OPENXAE_IR_H

#include "Operator.h"

#include <algorithm>
#include <map>
#include <memory>

#if BUILD_TORCH2PNNX
namespace torch {
namespace jit {
struct Value;
struct Node;
}// namespace jit
}// namespace torch
namespace at {
class Tensor;
}
#endif// BUILD_TORCH2PNNX

#if BUILD_ONNX2PNNX
namespace onnx {
class AttributeProto;
class TensorProto;
class ValueInfoProto;
}// namespace onnx
namespace pnnx {
namespace onnx2pnnx {
class OnnxAttributeProxy;
}// namespace onnx2pnnx
}// namespace pnnx
#endif// BUILD_ONNX2PNNX

namespace pnnx {

/// @deprecated
class Parameter_Deprecated {
public:
    /**
     * @brief Default constructor.
     */
    Parameter_Deprecated() : type_(ParameterType::kParameterUnknown) {}

    /**
     * @brief Constructor for bool type parameter.
     * @param val bool type value.
     */
    explicit Parameter_Deprecated(bool val)
        : type_(ParameterType::kParameterBool), boolVal_(val) {}

    /**
     * @brief Constructor for int type parameter.
     * @param val int type value.
     */
    explicit Parameter_Deprecated(int val)
        : type_(ParameterType::kParameterInt), intVal_(val) {}

    /**
     * @brief Constructor for long type parameter.
     * @param val long type value.
     */
    explicit Parameter_Deprecated(long val) : type_(ParameterType::kParameterInt) {
        if (val == std::numeric_limits<long>::min()) {
            val = std::numeric_limits<int>::min();
        }

        if (val == std::numeric_limits<long>::max()) {
            val = std::numeric_limits<int>::max();
        }
        intVal_ = static_cast<int>(val);
    }

    /**
     * @brief Constructor for long long type parameter.
     * @param val long long type value.
     */
    explicit Parameter_Deprecated(long long val) : type_(ParameterType::kParameterInt) {
        if (val == std::numeric_limits<long long>::min()) {
            val = std::numeric_limits<int>::min();
        }

        if (val == std::numeric_limits<long long>::max()) {
            val = std::numeric_limits<int>::max();
        }
        intVal_ = static_cast<int>(val);
    }

    /**
     * @brief Constructor for float type parameter.
     * @param val float type value.
     */
    explicit Parameter_Deprecated(float val)
        : type_(ParameterType::kParameterFloat), floatVal_(val) {}

    /**
     * @brief Constructor for double type parameter.
     * @param val double type value.
     */
    explicit Parameter_Deprecated(double val)
        : type_(ParameterType::kParameterFloat), floatVal_(static_cast<float>(val)) {}

    /**
     * @brief Constructor for string type parameter.
     * @param val string type value.
     */
    explicit Parameter_Deprecated(const char* val)
        : type_(ParameterType::kParameterString), strVal_(val) {}

    /**
     * @brief Constructor for string type parameter.
     * @param val string type value.
     */
    explicit Parameter_Deprecated(std::string val)
        : type_(ParameterType::kParameterString), strVal_(std::move(val)) {}

    /**
     * @brief Constructor for array int type parameter.
     * @param val init list of int type value.
     */
    Parameter_Deprecated(const std::initializer_list<int>& val)
        : type_(ParameterType::kParameterArrayInt), arrayIntVal_(val) {}

    /**
     * @brief Constructor for array int type parameter.
     * @param val init list of int64 type value.
     */
    Parameter_Deprecated(const std::initializer_list<int64_t>& val) : type_(ParameterType::kParameterArrayInt) {
        for (const auto& x: val) {
            int64_t l = x;
            if (l == std::numeric_limits<long>::min()) {
                l = std::numeric_limits<int>::min();
            }

            if (l == std::numeric_limits<long>::max()) {
                l = std::numeric_limits<int>::max();
            }
            arrayIntVal_.push_back(static_cast<int>(l));
        }
    }

    /**
     * @brief Constructor for array int type parameter.
     * @param val vector of int type value.
     */
    explicit Parameter_Deprecated(const std::vector<int>& val)
        : type_(ParameterType::kParameterArrayInt), arrayIntVal_(val) {}

    /**
     * @brief Constructor for array int64 type parameter.
     * @param val vector of int64 type value.
     */
    explicit Parameter_Deprecated(const std::vector<int64_t>& val) : type_(ParameterType::kParameterArrayInt) {
        for (const auto& x: val) {
            int64_t l = x;
            if (l == std::numeric_limits<long>::min()) {
                l = std::numeric_limits<int>::min();
            }

            if (l == std::numeric_limits<long>::max()) {
                l = std::numeric_limits<int>::max();
            }
            arrayIntVal_.push_back(static_cast<int>(l));
        }
    }

    /**
     * @brief Constructor for array float type parameter.
     * @param val init list of float type value.
     */
    Parameter_Deprecated(const std::initializer_list<float>& val)
        : type_(ParameterType::kParameterArrayFloat), arrayFloatVal_(val) {}

    /**
     * @brief Constructor for array double type parameter.
     * @param val init list of double type value.
     */
    Parameter_Deprecated(const std::initializer_list<double>& val)
        : type_(ParameterType::kParameterArrayFloat) {
        for (const auto& x: val) {
            arrayFloatVal_.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array float type parameter.
     * @param val vector of float type value.
     */
    explicit Parameter_Deprecated(const std::vector<float>& val)
        : type_(ParameterType::kParameterArrayFloat), arrayFloatVal_(val) {}

    /**
     * @brief Constructor for array float type parameter.
     * @param val vector of double type value.
     */
    explicit Parameter_Deprecated(const std::vector<double>& val)
        : type_(ParameterType::kParameterArrayFloat) {
        for (const auto& x: val) {
            arrayFloatVal_.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array string type parameter.
     * @param val init list of string type value.
     */
    Parameter_Deprecated(const std::initializer_list<const char*>& val)
        : type_(ParameterType::kParameterArrayString) {
        for (const auto& x: val) {
            arrayStringVal_.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array string type parameter.
     * @param val init list of string type value.
     */
    Parameter_Deprecated(const std::initializer_list<std::string>& val)
        : type_(ParameterType::kParameterArrayString), arrayStringVal_(val) {}

    /**
     * @brief Constructor for array string type parameter.
     * @param val vector of string type value.
     */
    explicit Parameter_Deprecated(const std::vector<std::string>& val)
        : type_(ParameterType::kParameterArrayString), arrayStringVal_(val) {}

    /**
     * @brief Constructor for complex type parameter.
     * @param val complex type value.
     */
    explicit Parameter_Deprecated(const std::complex<float>& val)
        : type_(ParameterType::kParameterComplex), complexVal_(val) {}

    /**
     * @brief Constructor for complex type parameter.
     * @param val complex type value.
     */
    explicit Parameter_Deprecated(const std::complex<double>& val)
        : type_(ParameterType::kParameterComplex), complexVal_(val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param val init list of complex type value.
     */
    Parameter_Deprecated(const std::initializer_list<std::complex<float>>& val)
        : type_(ParameterType::kParameterArrayComplex), arrayComplexVal_(val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param val init list of complex type value.
     */
    Parameter_Deprecated(const std::initializer_list<std::complex<double>>& val)
        : type_(ParameterType::kParameterArrayComplex) {
        for (const auto& x: val) {
            arrayComplexVal_.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array complex type parameter.
     * @param val vector of complex type value.
     */
    explicit Parameter_Deprecated(const std::vector<std::complex<float>>& val)
        : type_(ParameterType::kParameterArrayComplex), arrayComplexVal_(val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param val vector of complex type value.
     */
    explicit Parameter_Deprecated(const std::vector<std::complex<double>>& val)
        : type_(ParameterType::kParameterArrayComplex) {
        for (const auto& x: val) {
            arrayComplexVal_.emplace_back(x);
        }
    }

#if BUILD_TORCH2PNNX
    Parameter_Deprecated(const torch::jit::Node* value_node);
    Parameter_Deprecated(const torch::jit::Value* value);
#endif// BUILD_TORCH2PNNX
#if BUILD_ONNX2PNNX
    Parameter_Deprecated(const onnx::AttributeProto& attr);
    Parameter_Deprecated(const onnx2pnnx::OnnxAttributeProxy& attr);
#endif// BUILD_ONNX2PNNX

    Parameter_Deprecated(const Parameter_Deprecated&) = default;
    Parameter_Deprecated(Parameter_Deprecated&&) = default;
    Parameter_Deprecated& operator=(const Parameter_Deprecated&) = default;
    Parameter_Deprecated& operator=(Parameter_Deprecated&&) = default;

    NODISCARD const ParameterType& type() const {
        return type_;
    }

    void SetType(ParameterType type) {
        type_ = type;
    }

    NODISCARD bool toBool() const {
        return boolVal_;
    }

    NODISCARD int toInt() const {
        return intVal_;
    }

    NODISCARD float toFloat() const {
        return floatVal_;
    }

    NODISCARD std::complex<float> toComplex() const {
        return complexVal_;
    }

    NODISCARD const std::string& toString() const {
        return strVal_;
    }

    NODISCARD const std::vector<int>& toArrayInt() const {
        return arrayIntVal_;
    }

    NODISCARD const std::vector<float>& toArrayFloat() const {
        return arrayFloatVal_;
    }

    NODISCARD const std::vector<std::complex<float>>& toArrayComplex() const {
        return arrayComplexVal_;
    }

    NODISCARD const std::vector<std::string>& toArrayString() const {
        return arrayStringVal_;
    }

    void SetBoolValue(bool val) {
        boolVal_ = val;
    }

    void SetIntValue(int val) {
        intVal_ = val;
    }

    void SetLongValue(long val) {
        if (val == std::numeric_limits<long>::min()) {
            val = std::numeric_limits<int>::min();
        }

        if (val == std::numeric_limits<long>::max()) {
            val = std::numeric_limits<int>::max();
        }
        intVal_ = static_cast<int>(val);
    }

    void SetLongLongValue(long long val) {
        if (val == std::numeric_limits<long long>::min()) {
            val = std::numeric_limits<int>::min();
        }

        if (val == std::numeric_limits<long long>::max()) {
            val = std::numeric_limits<int>::max();
        }
        intVal_ = static_cast<int>(val);
    }

    void SetFloatValue(float val) {
        floatVal_ = val;
    }

    void SetFloatValue(double val) {
        floatVal_ = static_cast<float>(val);
    }

    void SetComplexValue(std::complex<float> val) {
        complexVal_ = val;
    }

    void SetStringValue(std::string val) {
        strVal_ = std::move(val);
    }

    void SetArrayInt(std::vector<int> val) {
        arrayIntVal_ = std::move(val);
    }

    void SetArrayFloat(std::vector<float> val) {
        arrayFloatVal_ = std::move(val);
    }

    void SetArrayString(std::vector<std::string> val) {
        arrayStringVal_ = std::move(val);
    }

    void SetArrayComplex(std::vector<std::complex<float>> val) {
        arrayComplexVal_ = std::move(val);
    }

    void AddElemToArrayInt(int val) {
        arrayIntVal_.push_back(val);
    }

    void AddElemToArrayFloat(float val) {
        arrayFloatVal_.push_back(val);
    }

    void AddElemToArrayComplex(std::complex<float> val) {
        arrayComplexVal_.push_back(val);
    }

    void AddElemToArrayString(std::string val) {
        arrayStringVal_.push_back(std::move(val));
    }

    static std::string Encode2String(const Parameter_Deprecated& param);

    static Parameter_Deprecated ParseFromString(const std::string& value);

private:
    /**
     * @brief Parameter_Deprecated type
     *
     * 0 = null \n
     * 1 = bool \n
     * 2 = int \n
     * 3 = float \n
     * 4 = string \n
     * 5 = array int \n
     * 6 = array float \n
     * 7 = array string \n
     * 8 = others \n
     * 10 = complex \n
     * 11 = array complex
     */
    ParameterType type_;

    bool boolVal_{};
    int intVal_{};
    float floatVal_{};
    std::complex<float> complexVal_;
    std::vector<int> arrayIntVal_;
    std::vector<float> arrayFloatVal_;
    std::vector<std::complex<float>> arrayComplexVal_;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string strVal_;
    std::vector<std::string> arrayStringVal_;
};

bool operator==(const Parameter_Deprecated& lhs, const Parameter_Deprecated& rhs);

}// namespace pnnx


#endif//OPENXAE_IR_H
