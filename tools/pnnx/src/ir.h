//
// Created by 赵丹 on 24-5-30.
//

#ifndef OPENXAE_IR_H
#define OPENXAE_IR_H

#include "utils.h"
#include <algorithm>
#include <climits>
#include <complex>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

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

/**
 * @brief Runtime parameter type_.
 *
 * Enumerates the parameter type_ supported for workload.
 */
enum class ParameterType {
    kParameterUnknown = 0,
    kParameterBool = 1,
    kParameterInt = 2,
    kParameterFloat = 3,
    kParameterString = 4,
    kParameterArrayInt = 5,
    kParameterArrayFloat = 6,
    kParameterArrayString = 7,
    kParameterComplex = 10,
    kParameterArrayComplex = 11
};

/**
 * @brief Runtime data type_.
 *
 * Enumerates the data type_ supported for workload.
 */
enum class DataType {
    kDataTypeUnknown = 0,
    kDataTypeFloat32 = 1,
    kDataTypeFloat64 = 2,
    kDataTypeFloat16 = 3,
    kDataTypeInt32 = 4,
    kDataTypeInt64 = 5,
    kDataTypeInt16 = 6,
    kDataTypeInt8 = 7,
    kDataTypeUInt8 = 8,
    kDataTypeBool = 9,
    kDataTypeComplex64 = 10,
    kDataTypeComplex128 = 11,
    kDataTypeComplex32 = 12,
    kDataTypeBFloat16 = 13
};


class Parameter {
public:
    /**
     * @brief Default constructor.
     */
    Parameter() : type_(ParameterType::kParameterUnknown) {}

    /**
     * @brief Constructor for bool type_ parameter.
     * @param val bool type_ value.
     */
    explicit Parameter(bool val)
        : type_(ParameterType::kParameterBool), boolVal_(val) {}

    /**
     * @brief Constructor for int type_ parameter.
     * @param val int type_ value.
     */
    explicit Parameter(int val)
        : type_(ParameterType::kParameterInt), intVal_(val) {}

    /**
     * @brief Constructor for long type_ parameter.
     * @param val long type_ value.
     */
    explicit Parameter(long val) : type_(ParameterType::kParameterInt) {
        if (val == std::numeric_limits<long>::min()) {
            val = std::numeric_limits<int>::min();
        }

        if (val == std::numeric_limits<long>::max()) {
            val = std::numeric_limits<int>::max();
        }
        intVal_ = static_cast<int>(val);
    }

    /**
     * @brief Constructor for long long type_ parameter.
     * @param val long long type_ value.
     */
    explicit Parameter(long long val) : type_(ParameterType::kParameterInt) {
        if (val == std::numeric_limits<long long>::min()) {
            val = std::numeric_limits<int>::min();
        }

        if (val == std::numeric_limits<long long>::max()) {
            val = std::numeric_limits<int>::max();
        }
        intVal_ = static_cast<int>(val);
    }

    /**
     * @brief Constructor for float type_ parameter.
     * @param val float type_ value.
     */
    explicit Parameter(float val)
        : type_(ParameterType::kParameterFloat), floatVal_(val) {}

    /**
     * @brief Constructor for double type_ parameter.
     * @param val double type_ value.
     */
    explicit Parameter(double val)
        : type_(ParameterType::kParameterFloat), floatVal_(static_cast<float>(val)) {}

    /**
     * @brief Constructor for string type_ parameter.
     * @param val string type_ value.
     */
    explicit Parameter(const char* val)
        : type_(ParameterType::kParameterString), strVal_(val) {}

    /**
     * @brief Constructor for string type_ parameter.
     * @param val string type_ value.
     */
    explicit Parameter(std::string val)
        : type_(ParameterType::kParameterString), strVal_(std::move(val)) {}

    /**
     * @brief Constructor for array int type_ parameter.
     * @param val init list of int type_ value.
     */
    Parameter(const std::initializer_list<int>& val)
        : type_(ParameterType::kParameterArrayInt), arrayIntVal_(val) {}

    /**
     * @brief Constructor for array int type_ parameter.
     * @param val init list of int64 type_ value.
     */
    Parameter(const std::initializer_list<int64_t>& val) : type_(ParameterType::kParameterArrayInt) {
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
     * @brief Constructor for array int type_ parameter.
     * @param val vector of int type_ value.
     */
    explicit Parameter(const std::vector<int>& val)
        : type_(ParameterType::kParameterArrayInt), arrayIntVal_(val) {}

    /**
     * @brief Constructor for array int64 type_ parameter.
     * @param val vector of int64 type_ value.
     */
    explicit Parameter(const std::vector<int64_t>& val) : type_(ParameterType::kParameterArrayInt) {
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
     * @brief Constructor for array float type_ parameter.
     * @param val init list of float type_ value.
     */
    Parameter(const std::initializer_list<float>& val)
        : type_(ParameterType::kParameterArrayFloat), arrayFloatVal_(val) {}

    /**
     * @brief Constructor for array double type_ parameter.
     * @param val init list of double type_ value.
     */
    Parameter(const std::initializer_list<double>& val)
        : type_(ParameterType::kParameterArrayFloat) {
        for (const auto& x: val) {
            arrayFloatVal_.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array float type_ parameter.
     * @param val vector of float type_ value.
     */
    explicit Parameter(const std::vector<float>& val)
        : type_(ParameterType::kParameterArrayFloat), arrayFloatVal_(val) {}

    /**
     * @brief Constructor for array float type_ parameter.
     * @param val vector of double type_ value.
     */
    explicit Parameter(const std::vector<double>& val)
        : type_(ParameterType::kParameterArrayFloat) {
        for (const auto& x: val) {
            arrayFloatVal_.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array string type_ parameter.
     * @param val init list of string type_ value.
     */
    Parameter(const std::initializer_list<const char*>& val)
        : type_(ParameterType::kParameterArrayString) {
        for (const auto& x: val) {
            arrayStringVal_.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array string type_ parameter.
     * @param val init list of string type_ value.
     */
    Parameter(const std::initializer_list<std::string>& val)
        : type_(ParameterType::kParameterArrayString), arrayStringVal_(val) {}

    /**
     * @brief Constructor for array string type_ parameter.
     * @param val vector of string type_ value.
     */
    explicit Parameter(const std::vector<std::string>& val)
        : type_(ParameterType::kParameterArrayString), arrayStringVal_(val) {}

    /**
     * @brief Constructor for complex type_ parameter.
     * @param val complex type_ value.
     */
    explicit Parameter(const std::complex<float>& val)
        : type_(ParameterType::kParameterComplex), complexVal_(val) {}

    /**
     * @brief Constructor for complex type_ parameter.
     * @param val complex type_ value.
     */
    explicit Parameter(const std::complex<double>& val)
        : type_(ParameterType::kParameterComplex), complexVal_(val) {}

    /**
     * @brief Constructor for array complex type_ parameter.
     * @param val init list of complex type_ value.
     */
    Parameter(const std::initializer_list<std::complex<float>>& val)
        : type_(ParameterType::kParameterArrayComplex), arrayComplexVal_(val) {}

    /**
     * @brief Constructor for array complex type_ parameter.
     * @param val init list of complex type_ value.
     */
    Parameter(const std::initializer_list<std::complex<double>>& val)
        : type_(ParameterType::kParameterArrayComplex) {
        for (const auto& x: val) {
            arrayComplexVal_.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array complex type_ parameter.
     * @param val vector of complex type_ value.
     */
    explicit Parameter(const std::vector<std::complex<float>>& val)
        : type_(ParameterType::kParameterArrayComplex), arrayComplexVal_(val) {}

    /**
     * @brief Constructor for array complex type_ parameter.
     * @param val vector of complex type_ value.
     */
    explicit Parameter(const std::vector<std::complex<double>>& val)
        : type_(ParameterType::kParameterArrayComplex) {
        for (const auto& x: val) {
            arrayComplexVal_.emplace_back(x);
        }
    }

#if BUILD_TORCH2PNNX
    Parameter(const torch::jit::Node* value_node);
    Parameter(const torch::jit::Value* value);
#endif// BUILD_TORCH2PNNX
#if BUILD_ONNX2PNNX
    Parameter(const onnx::AttributeProto& attr);
    Parameter(const onnx2pnnx::OnnxAttributeProxy& attr);
#endif// BUILD_ONNX2PNNX

    Parameter(const Parameter&) = default;
    Parameter(Parameter&&) = default;
    Parameter& operator=(const Parameter&) = default;
    Parameter& operator=(Parameter&&) = default;

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

    static std::string Encode2String(const Parameter& param);

    static Parameter ParseFromString(const std::string& value);

private:
    /**
     * @brief Parameter type_
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

bool operator==(const Parameter& lhs, const Parameter& rhs);

class Attribute {
public:
    /**
     * @brief Default constructor.
     */
    Attribute() : type_(DataType::kDataTypeUnknown) {}

    Attribute(const std::initializer_list<int>& shape, const std::vector<float>& t);

#if BUILD_TORCH2PNNX
    Attribute(const at::Tensor& t);
#endif
#if BUILD_ONNX2PNNX
    Attribute(const onnx::TensorProto& t);
#endif

    Attribute(const Attribute&) = default;
    Attribute(Attribute&&) = default;
    Attribute& operator=(const Attribute&) = delete;
    Attribute& operator=(Attribute&&) = delete;

    NODISCARD const DataType& type() const {
        return type_;
    }

    void SetType(DataType type) {
        type_ = type;
    }

    NODISCARD size_t GetElemSize() const;

    NODISCARD size_t size() const;

    // convenient routines for manipulate fp16/fp32 weight
    NODISCARD std::vector<float> get_float32_data() const;

    void set_float32_data(const std::vector<float>& data_);

    /**
     * @brief Runtime data type.
     *
     * 0 = null \n
     * 1 = float32 \n
     * 2 = float64 \n
     * 3 = float16 \n
     * 4 = int32 \n
     * 5 = int64 \n
     * 6 = int16 \n
     * 7 = int8 \n
     * 8 = uint8 \n
     * 9 = bool \n
     * 10 = complex64 \n
     * 11 = complex128 \n
     * 12 = complex32 \n
     * 13 = bf16
     */

    std::vector<int> shape_;
    std::vector<char> data_;
    std::map<std::string, Parameter> params;

private:
    DataType type_;
};

bool operator==(const Attribute& lhs, const Attribute& rhs);

/**
 * @brief Concat two attributes along the first axis.
 * @param a left attribute
 * @param b right attribute
 * @return new attribute object.
 */
Attribute operator+(const Attribute& a, const Attribute& b);

class Operator;
class Operand {
public:
    void remove_consumer(const Operator*);
    Operator* producer{};
    std::vector<Operator*> consumers;
    /**
     * @brief Runtime data type_.
     *
     * 0 = null \n
     * 1 = float32 \n
     * 2 = float64 \n
     * 3 = float16 \n
     * 4 = int32 \n
     * 5 = int64 \n
     * 6 = int16 \n
     * 7 = int8 \n
     * 8 = uint8 \n
     * 9 = bool \n
     * 10 = complex64 \n
     * 11 = complex128 \n
     * 12 = complex32 \n
     * 13 = bf16
     */
    DataType type;
    std::vector<int> shape;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string name;
    std::map<std::string, Parameter> params;

private:
    Operand() : type(DataType::kDataTypeUnknown) {}
    friend class Graph;
};

class Operator {
public:
    bool has_param(const std::string& key) const {
        return params.find(key) != params.end();
    }

    bool has_attr(const std::string& key) const {
        return attrs.find(key) != attrs.end();
    }

    bool has_input(const std::string& key) const {
        return std::find(input_names.begin(), input_names.end(), key) != input_names.end();
    }

    Operand* named_input(const std::string& key);
    const Operand* named_input(const std::string& key) const;

    std::string GetOpType() const {
        return type;
    }

    std::string GetOpName() const {
        return name;
    }

    std::vector<Operand*> inputs;
    std::vector<Operand*> outputs;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string type;
    std::string name;
    std::vector<std::string> input_names;
    std::map<std::string, Parameter> params;
    std::map<std::string, Attribute> attrs;

private:
    friend class Graph;
    Operator() = default;
};

class Graph {
public:
    /**
     * @brief Default constructor.
     */
    Graph() = default;

    Graph(const Graph&) = delete;

    Graph(Graph&&) = delete;

    Graph& operator=(const Graph&) = delete;

    Graph& operator=(Graph&&) = delete;

    /**
     * @brief Destructor.
     */
    ~Graph();

    int load(const std::string& paramPath, const std::string& binPath);

    int save(const std::string& paramPath, const std::string& binPath);

    int python(const std::string& pyPath, const std::string& binPath);

    int parse(const std::string& param);

private:
    Operator* CreateOperator(const std::string& type, const std::string& name);

    Operator* CreateOperatorBefore(const std::string& type, const std::string& name, const Operator* cur);

    Operator* CreateOperatorAfter(const std::string& type, const std::string& name, const Operator* cur);

    Operand* CreateOperand(const std::string& name);

    Operand* GetOperand(const std::string& name);

    const Operand* GetOperand(const std::string& name) const;

#if BUILD_TORCH2PNNX
    Operand* new_operand(const torch::jit::Value* v);
#endif

#if BUILD_ONNX2PNNX
    Operand* new_operand(const onnx::ValueInfoProto& value);
    Operand* new_operand(const onnx::TensorProto& t);
#endif

    std::vector<Operator*> ops;
    std::vector<Operand*> operands;

    std::vector<std::shared_ptr<Operator>> ops_;
};

}// namespace pnnx


#endif//OPENXAE_IR_H
