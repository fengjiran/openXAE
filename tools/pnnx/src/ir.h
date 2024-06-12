//
// Created by 赵丹 on 24-5-30.
//

#ifndef OPENXAE_IR_H
#define OPENXAE_IR_H

#include "utils.h"

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
 * @brief Runtime parameter type.
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
 * @brief Runtime data type.
 *
 * Enumerates the data type supported for workload.
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

using param_null_type = std::integral_constant<ParameterType, ParameterType::kParameterUnknown>;
using param_bool_type = std::integral_constant<ParameterType, ParameterType::kParameterBool>;
using param_int_type = std::integral_constant<ParameterType, ParameterType::kParameterInt>;
using param_float_type = std::integral_constant<ParameterType, ParameterType::kParameterFloat>;
using param_complex_type = std::integral_constant<ParameterType, ParameterType::kParameterComplex>;
using param_string_type = std::integral_constant<ParameterType, ParameterType::kParameterString>;
using param_arrayint_type = std::integral_constant<ParameterType, ParameterType::kParameterArrayInt>;
using param_arrayfloat_type = std::integral_constant<ParameterType, ParameterType::kParameterArrayFloat>;
using param_arraycomplex_type = std::integral_constant<ParameterType, ParameterType::kParameterArrayComplex>;
using param_arraystring_type = std::integral_constant<ParameterType, ParameterType::kParameterArrayString>;

template<typename T, typename = void>
struct GetParameterType : param_null_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_same_v<T, bool>>>
    : param_bool_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
    : param_int_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_floating_point_v<T>>>
    : param_float_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_same_v<std::decay_t<T>, std::complex<float>>>>
    : param_complex_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_string_v<T>>>
    : param_string_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_int_v<T>>>
    : param_arrayint_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_float_v<T>>>
    : param_arrayfloat_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_string_v<T>>>
    : param_arraystring_type {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_complex_v<T>>>
    : param_arraycomplex_type {};

template<typename T>
ParameterType get_parameter_type() {
    if (std::is_same_v<T, bool>) {
        return ParameterType::kParameterBool;
    }

    if (std::is_integral_v<T>) {
        return ParameterType::kParameterInt;
    }

    if (std::is_floating_point_v<T>) {
        return ParameterType::kParameterFloat;
    }

    if (std::is_same_v<std::decay_t<T>, std::complex<float>>) {
        return ParameterType::kParameterComplex;
    }

    if (is_string_v<T>) {
        return ParameterType::kParameterString;
    }

    if (is_std_vector_int_v<T>) {
        return ParameterType::kParameterArrayInt;
    }

    if (is_std_vector_float_v<T>) {
        return ParameterType::kParameterArrayFloat;
    }

    if (is_std_vector_string_v<T>) {
        return ParameterType::kParameterArrayString;
    }

    if (is_std_vector_complex_v<T>) {
        return ParameterType::kParameterArrayComplex;
    }

    return ParameterType::kParameterUnknown;
}


template<typename T>
class Parameter_ {
public:
    using value_type = T;

    Parameter_() : type_(GetParameterType<T>::value) {}

    explicit Parameter_(T val)
        : type_(GetParameterType<T>::value),
          value_(val) {}

    NODISCARD const ParameterType& type() const {
        return type_;
    }

    void SetType(ParameterType type) {
        type_ = type;
    }

    T toValue() const {
        return value_;
    }

    template<typename U,
             typename std::enable_if<std::is_convertible_v<U, T>>::type* = nullptr>
    void SetValue(U&& value) {
        value_ = std::forward<U>(value);
    }

    /**
     * @brief Encode unknown type parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterUnknown>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return "None";
    }

    /**
     * @brief Encode bool parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterBool>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return toValue() ? "True" : "False";
    }

    /**
     * @brief Encode int parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterInt>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return std::to_string(toValue());
    }

    /**
     * @brief Encode float parameter to string.
     * @tparam U
     * @param param
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterFloat>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        char buf[64];
        snprintf(buf, sizeof(buf), "%e", toValue());
        return buf;
    }

    /**
     * @brief Encode string parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterString>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return toValue();
    }

    /**
     * @brief Encode complex parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterComplex>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        char buf[128];
        snprintf(buf, sizeof(buf), "%e+%ei", toValue().real(), toValue().imag());
        return buf;
    }

private:
    /**
     * @brief Parameter type
     */
    ParameterType type_;

    T value_{};
};

template<typename T>
class Parameter_<std::vector<T>> {
public:
    using value_type = std::vector<T>;

    Parameter_() : type_(GetParameterType<std::vector<T>>::value) {}

    explicit Parameter_(const std::vector<T>& val)
        : type_(GetParameterType<std::vector<T>>::value), value_(val) {}

    NODISCARD const ParameterType& type() const {
        return type_;
    }

    void SetType(ParameterType type) {
        type_ = type;
    }

    std::vector<T> toValue() const {
        return value_;
    }

    void SetValue(std::vector<T> value) {
        value_ = std::move(value);
    }

    template<typename U,
             typename std::enable_if<std::is_convertible_v<U, T>>::type* = nullptr>
    void AddElemToArray(U&& value) {
        value_.push_back(std::forward<U>(value));
    }

    /**
     * @brief Encode array int parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = std::vector<T>,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterArrayInt>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        std::string code;
        code += "(";
        size_t size = toValue().size();
        for (const auto& ele: toValue()) {
            code += (std::to_string(ele) + (--size ? "," : ""));
        }
        code += ")";
        return code;
    }

    /**
     * @brief Encode array float parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = std::vector<T>,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterArrayFloat>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        std::string code;
        code += "(";
        size_t size = toValue().size();
        for (const auto& ele: toValue()) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%e", ele);
            code += (std::string(buf) + (--size ? "," : ""));
        }
        code += ")";
        return code;
    }

    /**
     * @brief Encode array string parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = std::vector<T>,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterArrayString>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        std::string code;
        code += "(";
        size_t size = toValue().size();
        for (const auto& ele: toValue()) {
            code += (ele + (--size ? "," : ""));
        }
        code += ")";
        return code;
    }

    /**
     * @brief Encode array complex parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = std::vector<T>,
             typename std::enable_if<GetParameterType<U>::value == ParameterType::kParameterArrayComplex>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        std::string code;
        code += "(";
        size_t size = toValue().size();
        for (const auto& ele: toValue()) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%e+%ei", ele.real(), ele.imag());
            code += (std::string(buf) + (--size ? "," : ""));
        }
        code += ")";
        return code;
    }

private:
    ParameterType type_;
    std::vector<T> value_;
};

// CTAD Deduction Guides
Parameter_(const char*) -> Parameter_<std::string>;

Parameter_(std::initializer_list<const char*>) -> Parameter_<std::vector<std::string>>;

Parameter_(std::vector<const char*>) -> Parameter_<std::vector<std::string>>;

template<typename T>
Parameter_(std::initializer_list<T>) -> Parameter_<std::vector<T>>;

template<typename T>
bool operator==(const Parameter_<T>& lhs, const Parameter_<T>& rhs) {
    if (lhs.type() != rhs.type()) {
        return false;
    }

    if (lhs.type() == ParameterType::kParameterUnknown) {
        return true;
    }

    return lhs.toValue() == rhs.toValue();
}

template<typename... Args>
auto make_parameter(Args&&... args) {
    return Parameter_<Args...>(std::forward<Args>(args)...);
}

class Parameter {
public:
    /**
     * @brief Default constructor.
     */
    Parameter() : type_(ParameterType::kParameterUnknown) {}

    /**
     * @brief Constructor for bool type parameter.
     * @param val bool type value.
     */
    explicit Parameter(bool val)
        : type_(ParameterType::kParameterBool), boolVal_(val) {}

    /**
     * @brief Constructor for int type parameter.
     * @param val int type value.
     */
    explicit Parameter(int val)
        : type_(ParameterType::kParameterInt), intVal_(val) {}

    /**
     * @brief Constructor for long type parameter.
     * @param val long type value.
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
     * @brief Constructor for long long type parameter.
     * @param val long long type value.
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
     * @brief Constructor for float type parameter.
     * @param val float type value.
     */
    explicit Parameter(float val)
        : type_(ParameterType::kParameterFloat), floatVal_(val) {}

    /**
     * @brief Constructor for double type parameter.
     * @param val double type value.
     */
    explicit Parameter(double val)
        : type_(ParameterType::kParameterFloat), floatVal_(static_cast<float>(val)) {}

    /**
     * @brief Constructor for string type parameter.
     * @param val string type value.
     */
    explicit Parameter(const char* val)
        : type_(ParameterType::kParameterString), strVal_(val) {}

    /**
     * @brief Constructor for string type parameter.
     * @param val string type value.
     */
    explicit Parameter(std::string val)
        : type_(ParameterType::kParameterString), strVal_(std::move(val)) {}

    /**
     * @brief Constructor for array int type parameter.
     * @param val init list of int type value.
     */
    Parameter(const std::initializer_list<int>& val)
        : type_(ParameterType::kParameterArrayInt), arrayIntVal_(val) {}

    /**
     * @brief Constructor for array int type parameter.
     * @param val init list of int64 type value.
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
     * @brief Constructor for array int type parameter.
     * @param val vector of int type value.
     */
    explicit Parameter(const std::vector<int>& val)
        : type_(ParameterType::kParameterArrayInt), arrayIntVal_(val) {}

    /**
     * @brief Constructor for array int64 type parameter.
     * @param val vector of int64 type value.
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
     * @brief Constructor for array float type parameter.
     * @param val init list of float type value.
     */
    Parameter(const std::initializer_list<float>& val)
        : type_(ParameterType::kParameterArrayFloat), arrayFloatVal_(val) {}

    /**
     * @brief Constructor for array double type parameter.
     * @param val init list of double type value.
     */
    Parameter(const std::initializer_list<double>& val)
        : type_(ParameterType::kParameterArrayFloat) {
        for (const auto& x: val) {
            arrayFloatVal_.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array float type parameter.
     * @param val vector of float type value.
     */
    explicit Parameter(const std::vector<float>& val)
        : type_(ParameterType::kParameterArrayFloat), arrayFloatVal_(val) {}

    /**
     * @brief Constructor for array float type parameter.
     * @param val vector of double type value.
     */
    explicit Parameter(const std::vector<double>& val)
        : type_(ParameterType::kParameterArrayFloat) {
        for (const auto& x: val) {
            arrayFloatVal_.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array string type parameter.
     * @param val init list of string type value.
     */
    Parameter(const std::initializer_list<const char*>& val)
        : type_(ParameterType::kParameterArrayString) {
        for (const auto& x: val) {
            arrayStringVal_.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array string type parameter.
     * @param val init list of string type value.
     */
    Parameter(const std::initializer_list<std::string>& val)
        : type_(ParameterType::kParameterArrayString), arrayStringVal_(val) {}

    /**
     * @brief Constructor for array string type parameter.
     * @param val vector of string type value.
     */
    explicit Parameter(const std::vector<std::string>& val)
        : type_(ParameterType::kParameterArrayString), arrayStringVal_(val) {}

    /**
     * @brief Constructor for complex type parameter.
     * @param val complex type value.
     */
    explicit Parameter(const std::complex<float>& val)
        : type_(ParameterType::kParameterComplex), complexVal_(val) {}

    /**
     * @brief Constructor for complex type parameter.
     * @param val complex type value.
     */
    explicit Parameter(const std::complex<double>& val)
        : type_(ParameterType::kParameterComplex), complexVal_(val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param val init list of complex type value.
     */
    Parameter(const std::initializer_list<std::complex<float>>& val)
        : type_(ParameterType::kParameterArrayComplex), arrayComplexVal_(val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param val init list of complex type value.
     */
    Parameter(const std::initializer_list<std::complex<double>>& val)
        : type_(ParameterType::kParameterArrayComplex) {
        for (const auto& x: val) {
            arrayComplexVal_.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array complex type parameter.
     * @param val vector of complex type value.
     */
    explicit Parameter(const std::vector<std::complex<float>>& val)
        : type_(ParameterType::kParameterArrayComplex), arrayComplexVal_(val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param val vector of complex type value.
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
     * @brief Parameter type
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

    Attribute(const std::vector<int>& shape, const std::vector<float>& t);

#if BUILD_TORCH2PNNX
    Attribute(const at::Tensor& t);
#endif
#if BUILD_ONNX2PNNX
    Attribute(const onnx::TensorProto& t);
#endif

    Attribute(const Attribute&) = delete;
    Attribute(Attribute&& other) noexcept
        : type_(other.type_), data_(std::move(other.data_)), shape_(std::move(other.shape_)) {}

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

    NODISCARD const std::vector<int>& GetShape() const {
        return shape_;
    }

    std::vector<int>& GetShape() {
        return shape_;
    }

    NODISCARD const std::vector<char>& GetRawData() const {
        return data_;
    }

    std::vector<char>& GetRawData() {
        return data_;
    }

    // convenient routines for manipulate fp16/fp32 weight
    NODISCARD std::vector<float> CastToFloat32() const;

    void SetFloat32Data(const std::vector<float>& newData);

private:
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
    DataType type_;
    std::vector<int> shape_;
    std::vector<char> data_;
    //    std::map<std::string, Parameter> params;
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
    Operand() : type_(DataType::kDataTypeUnknown) {}

    explicit Operand(std::string name) : name_(std::move(name)), type_(DataType::kDataTypeUnknown) {}

    Operand(std::string name, DataType type, std::vector<int> shape)
        : name_(std::move(name)), type_(type), shape_(std::move(shape)) {}


    Operand(const Operand&) = delete;
    Operand(Operand&&) = delete;
    Operand& operator=(const Operand&) = delete;
    Operand& operator=(Operand&&) = delete;

    NODISCARD const DataType& type() const {
        return type_;
    }

    void SetType(DataType type) {
        type_ = type;
    }

    NODISCARD const std::vector<int>& GetShape() const {
        return shape_;
    }

    std::vector<int>& GetShape() {
        return shape_;
    }

    NODISCARD const std::string& name() const {
        return name_;
    }

    std::map<std::string, Parameter>& GetParams() {
        return params_;
    }

    NODISCARD const std::map<std::string, Parameter>& GetParams() const {
        return params_;
    }

    void SetProducer(const std::shared_ptr<Operator>& op) {
        producer_ = op;
    }

    void AddConsumer(const std::shared_ptr<Operator>& op) {
        consumers_.push_back(op);
    }

    NODISCARD const std::vector<std::shared_ptr<Operator>>& GetConsumers() const {
        return consumers_;
    }

    std::vector<std::shared_ptr<Operator>>& GetConsumers() {
        return consumers_;
    }

    void RemoveConsumer(const std::shared_ptr<Operator>& op);

private:
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
    DataType type_;
    std::vector<int> shape_;
    std::shared_ptr<Operator> producer_;
    std::vector<std::shared_ptr<Operator>> consumers_;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string name_;
    std::map<std::string, Parameter> params_;
};

class Operator {
public:
    Operator() = default;

    Operator(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}

    Operator(std::string name, std::string type, std::map<std::string, std::shared_ptr<Parameter>> params,
             std::map<std::string, std::shared_ptr<Attribute>> attrs, std::vector<std::shared_ptr<Operand>> inputOperands,
             std::vector<std::string> inputNames)
        : name_(std::move(name)), type_(std::move(type)), params_(std::move(params)), attrs_(std::move(attrs)),
          inputOperands_(std::move(inputOperands)), inputNames_(std::move(inputNames)) {}

    Operator(const Operator&) = delete;
    Operator(Operator&&) = delete;
    Operator& operator=(const Operator&) = delete;
    Operator& operator=(Operator&&) = delete;

    NODISCARD bool HasParam(const std::string& key) const {
        return params_.find(key) != params_.end();
    }

    NODISCARD bool HasAttr(const std::string& key) const {
        return attrs_.find(key) != attrs_.end();
    }

    NODISCARD bool HasInput(const std::string& key) const {
        return std::find(inputNames_.begin(), inputNames_.end(), key) != inputNames_.end();
    }

    NODISCARD std::shared_ptr<Operand> GetNamedInput(const std::string& key) const;

    NODISCARD const std::string& type() const {
        return type_;
    }

    NODISCARD const std::string& name() const {
        return name_;
    }

    NODISCARD const std::vector<std::string>& GetInputNames() const {
        return inputNames_;
    }

    std::vector<std::string>& GetInputNames() {
        return inputNames_;
    }

    NODISCARD const std::vector<std::shared_ptr<Operand>>& GetInputOperands() const {
        return inputOperands_;
    }

    NODISCARD const std::vector<std::shared_ptr<Operand>>& GetOutputOperands() const {
        return outputOperands_;
    }

    void AddInputOperand(const std::shared_ptr<Operand>& operand) {
        inputOperands_.push_back(operand);
    }

    void AddOutputOperand(const std::shared_ptr<Operand>& operand) {
        outputOperands_.push_back(operand);
    }

    NODISCARD const std::map<std::string, std::shared_ptr<Parameter>>& GetParameters() const {
        return params_;
    }

    std::map<std::string, std::shared_ptr<Parameter>>& GetParameters() {
        return params_;
    }

    NODISCARD const std::map<std::string, std::shared_ptr<Attribute>>& GetAttributes() const {
        return attrs_;
    }

    std::map<std::string, std::shared_ptr<Attribute>>& GetAttributes() {
        return attrs_;
    }

private:
    // keep std::string typed member the last for cross cxxabi compatibility
    std::string type_;
    std::string name_;

    std::vector<std::string> inputNames_;

    std::vector<std::shared_ptr<Operand>> inputOperands_;
    std::vector<std::shared_ptr<Operand>> outputOperands_;

    std::map<std::string, std::shared_ptr<Parameter>> params_;
    std::map<std::string, std::shared_ptr<Attribute>> attrs_;
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

    int load(const std::string& paramPath, const std::string& binPath);

    int save(const std::string& paramPath, const std::string& binPath);

    int python(const std::string& pyPath, const std::string& binPath);

    int parse(const std::string& param);

    std::shared_ptr<Operator> CreateOperator(const std::string& type, const std::string& name);

    std::shared_ptr<Operand> CreateOperator(const std::string& type, const std::string& name,
                                            const std::map<std::string, std::shared_ptr<Parameter>>& params,
                                            const std::map<std::string, std::shared_ptr<Attribute>>& attrs,
                                            const std::vector<std::shared_ptr<Operand>>& inputOperands,
                                            const std::vector<std::string>& inputOperandNames,
                                            const std::string& outputName,
                                            DataType outputType,
                                            const std::vector<int>& outputShape);

    std::shared_ptr<Operator> CreateOperatorBefore(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur);

    std::shared_ptr<Operator> CreateOperatorAfter(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur);

    std::shared_ptr<Operand> CreateOperand(const std::string& name);

    std::shared_ptr<Operand> GetOperand(const std::string& name);

#if BUILD_TORCH2PNNX
    Operand* new_operand(const torch::jit::Value* v);
#endif

#if BUILD_ONNX2PNNX
    Operand* new_operand(const onnx::ValueInfoProto& value);
    Operand* new_operand(const onnx::TensorProto& t);
#endif

private:
    std::vector<std::shared_ptr<Operator>> ops_;
    std::vector<std::shared_ptr<Operand>> operands_;
};

}// namespace pnnx


#endif//OPENXAE_IR_H
