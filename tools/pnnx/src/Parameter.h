//
// Created by richard on 6/13/24.
//

#ifndef OPENXAE_PARAMETER_H
#define OPENXAE_PARAMETER_H

#include "utils.h"

#include <complex>
#include <torch/script.h>
#include <variant>
#include <vector>

namespace pnnx {

template<typename T>
struct is_vector : std::false_type {
    using elem_type = T;
};

template<typename T, typename alloc>
struct is_vector<std::vector<T, alloc>> : std::true_type {
    using elem_type = T;
};

template<typename T>
struct is_vector<std::initializer_list<T>> : std::true_type {
    using elem_type = T;
};

template<typename T>
using is_std_vector =
        typename is_vector<std::remove_cv_t<std::remove_reference_t<T>>>::type;

template<typename T>
constexpr bool is_string_v =
        std::is_same_v<std::decay_t<T>, std::string> || std::is_convertible_v<T, std::string>;

template<typename T>
constexpr bool is_std_vector_v = is_std_vector<T>::value;

//
template<typename T>
constexpr bool is_std_vector_int_v = is_std_vector_v<T> && std::is_integral_v<typename is_vector<T>::elem_type>;

template<typename T>
constexpr bool is_std_vector_float_v = is_std_vector_v<T> && std::is_floating_point_v<typename is_vector<T>::elem_type>;

template<typename T>
constexpr bool is_std_vector_string_v = is_std_vector_v<T> && is_string_v<typename is_vector<T>::elem_type>;

template<typename T>
constexpr bool is_std_vector_complex_v = is_std_vector_v<T> && std::is_same_v<std::decay_t<typename is_vector<T>::elem_type>, std::complex<float>>;

template<typename T, typename = void>
struct GetParameterType
    : std::integral_constant<ParameterType, ParameterType::kParameterUnknown> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_same_v<T, bool>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterBool> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterInt> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_floating_point_v<T>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterFloat> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<std::is_same_v<std::decay_t<T>, std::complex<float>>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterComplex> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_string_v<T>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterString> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_int_v<T>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterArrayInt> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_float_v<T>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterArrayFloat> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_string_v<T>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterArrayString> {};

template<typename T>
struct GetParameterType<T, typename std::enable_if_t<is_std_vector_complex_v<T>>>
    : std::integral_constant<ParameterType, ParameterType::kParameterArrayComplex> {};


template<typename T>
class Parameter {
public:
    using value_type = T;

    Parameter() : type_(GetParameterType<T>()) {}

    explicit Parameter(T val)
        : type_(GetParameterType<T>()),
          value_(val) {}

#if BUILD_TORCH2PNNX
    explicit Parameter(const torch::jit::Node* value_node);
    explicit Parameter(const torch::jit::Value* value);
#endif// BUILD_TORCH2PNNX

#if BUILD_ONNX2PNNX
    Parameter(const onnx::AttributeProto& attr);
    Parameter(const onnx2pnnx::OnnxAttributeProxy& attr);
#endif// BUILD_ONNX2PNNX

    NODISCARD const ParameterType& type() const {
        return type_;
    }

    void SetType(ParameterType type) {
        type_ = type;
    }

    NODISCARD T toValue() const {
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterUnknown>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return "None";
    }

    /**
     * @brief Encode bool parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterBool>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return toValue() ? "True" : "False";
    }

    /**
     * @brief Encode int parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterInt>::type* = nullptr>
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterFloat>::type* = nullptr>
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterString>::type* = nullptr>
    NODISCARD std::string Encode2String() const {
        return toValue();
    }

    /**
     * @brief Encode complex parameter to string.
     * @tparam U
     * @return
     */
    template<typename U = T,
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterComplex>::type* = nullptr>
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

#if BUILD_TORCH2PNNX
template<typename T>
Parameter<T>::Parameter(const torch::jit::Node* value_node) {
    type_ = ParameterType::kParameterUnknown;
    if (value_node->kind() == c10::prim::Constant) {
        if (value_node->output()->type()->kind() == c10::TypeKind::NoneType) {
            type_ = ParameterType::kParameterUnknown;
            return;
        }

        if (!value_node->hasAttribute(torch::jit::attr::value)) {
            std::cerr << "No attribute value.\n";
            value_node->dump();
            return;
        }

        switch (value_node->output()->type()->kind()) {
            case c10::TypeKind::NoneType: {
                type_ = ParameterType::kParameterUnknown;
                break;
            }

            case c10::TypeKind::BoolType: {
                type_ = ParameterType::kParameterBool;
                value_ = (bool) value_node->i(torch::jit::attr::value);
                break;
            }

            case c10::TypeKind::IntType: {
                type_ = ParameterType::kParameterInt;
                int64_t i64 = value_node->i(torch::jit::attr::value);
                if (i64 == std::numeric_limits<int64_t>::max()) {
                    i64 = std::numeric_limits<int>::max();
                }

                if (i64 == std::numeric_limits<int64_t>::min()) {
                    i64 = std::numeric_limits<int>::min();
                }

                value_ = (int) i64;
                break;
            }

            case c10::TypeKind::FloatType: {
                type_ = ParameterType::kParameterFloat;
                value_ = (float) value_node->f(torch::jit::attr::value);
                break;
            }

            case c10::TypeKind::StringType:
            case c10::TypeKind::DeviceObjType: {
                type_ = ParameterType::kParameterString;
                value_ = std::string(value_node->s(torch::jit::attr::value));
                break;
            }

#if Torch_VERSION_MAJOR >= 2 || (Torch_VERSION_MAJOR >= 1 && Torch_VERSION_MINOR >= 9)
            case c10::TypeKind::ComplexType: {
                type_ = ParameterType::kParameterComplex;
                value_ = std::complex<float>(value_node->c(torch::jit::attr::value));
                break;
            }
#endif

            case c10::TypeKind::TensorType: {
                const at::Tensor& t = value_node->t(torch::jit::attr::value);
                if (t.dim() == 0 && t.numel() == 1) {
                    if (t.scalar_type() == c10::ScalarType::Long) {
                        type_ = ParameterType::kParameterInt;
                        int64_t i64 = value_node->i(torch::jit::attr::value);
                        if (i64 == std::numeric_limits<int64_t>::max()) {
                            i64 = std::numeric_limits<int>::max();
                        }

                        if (i64 == std::numeric_limits<int64_t>::min()) {
                            i64 = std::numeric_limits<int>::min();
                        }

                        value_ = (int) i64;
                    } else if (t.scalar_type() == c10::ScalarType::Int) {
                        type_ = ParameterType::kParameterInt;
                        value_ = t.item<int>();
                    } else if (t.scalar_type() == c10::ScalarType::Double) {
                        type_ = ParameterType::kParameterFloat;
                        value_ = (float) t.item<double>();
                    } else if (t.scalar_type() == c10::ScalarType::Float) {
                        type_ = ParameterType::kParameterFloat;
                        value_ = t.item<float>();
                    } else if (t.scalar_type() == c10::ScalarType::ComplexDouble) {
                        type_ = ParameterType::kParameterComplex;
                        value_ = std::complex<float>(t.item<c10::complex<double>>());
                    } else if (t.scalar_type() == c10::ScalarType::ComplexFloat) {
                        type_ = ParameterType::kParameterComplex;
                        value_ = std::complex<float>(t.item<c10::complex<float>>());
                    } else {
                        std::cerr << "Unknown Parameter value kind " << value_node->kind().toDisplayString()
                                  << " of TensorType, t.dim = 0\n";
                    }
                } else {
                    // constant tensor will become pnnx attribute node later.
                    type_ = ParameterType::kParameterOther;
                }
                break;
            }

            case c10::TypeKind::ListType: {

            }
        }

    } else if (value_node->kind() == c10::prim::ListConstruct) {
        //
    } else {
        std::cerr << "Unknown Parameter value_node kind "
                  << value_node->kind().toDisplayString();
    }
}

template<typename T>
Parameter<T>::Parameter(const torch::jit::Value* value)
    : Parameter(value->node()) {}

#endif


template<typename T>
class Parameter<std::vector<T>> {
public:
    using value_type = std::vector<T>;

    Parameter() : type_(GetParameterType<std::vector<T>>()) {}

    explicit Parameter(const std::vector<T>& val)
        : type_(GetParameterType<std::vector<T>>()), value_(val) {}

    NODISCARD const ParameterType& type() const {
        return type_;
    }

    void SetType(ParameterType type) {
        type_ = type;
    }

    NODISCARD std::vector<T> toValue() const {
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterArrayInt>::type* = nullptr>
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterArrayFloat>::type* = nullptr>
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterArrayString>::type* = nullptr>
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
             typename std::enable_if<GetParameterType<U>() == ParameterType::kParameterArrayComplex>::type* = nullptr>
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
Parameter(const char*) -> Parameter<std::string>;

Parameter(std::initializer_list<const char*>) -> Parameter<std::vector<std::string>>;

Parameter(std::vector<const char*>) -> Parameter<std::vector<std::string>>;

template<typename T>
Parameter(std::initializer_list<T>) -> Parameter<std::vector<T>>;

template<typename T>
bool operator==(const Parameter<T>& lhs, const Parameter<T>& rhs) {
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
    return Parameter<Args...>(std::forward<Args>(args)...);
}


using ParameterVar = std::variant<
        //        Parameter<void*>,
        Parameter<bool>,
        Parameter<int>,
        Parameter<float>,
        Parameter<double>,
        Parameter<std::string>,
        Parameter<std::complex<float>>,
        Parameter<std::vector<int>>,
        Parameter<std::vector<float>>,
        Parameter<std::vector<std::string>>>;

}// namespace pnnx

#endif//OPENXAE_PARAMETER_H
