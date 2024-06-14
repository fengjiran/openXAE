//
// Created by richard on 6/13/24.
//

#ifndef OPENXAE_PARAMETER_H
#define OPENXAE_PARAMETER_H

#include "utils.h"

#include <complex>
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
class Parameter {
public:
    using value_type = T;

    Parameter() : type_(GetParameterType<T>::value) {}

    explicit Parameter(T val)
        : type_(GetParameterType<T>::value),
          value_(val) {}

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
class Parameter<std::vector<T>> {
public:
    using value_type = std::vector<T>;

    Parameter() : type_(GetParameterType<std::vector<T>>::value) {}

    explicit Parameter(const std::vector<T>& val)
        : type_(GetParameterType<std::vector<T>>::value), value_(val) {}

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
