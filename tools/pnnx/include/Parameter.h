//
// Created by richard on 6/13/24.
//

#ifndef OPENXAE_PARAMETER_H
#define OPENXAE_PARAMETER_H

#include "utils.h"

#include <any>
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
constexpr bool is_string_v = std::is_convertible_v<T, std::string>;

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


class ParameterBase {
public:
    virtual ~ParameterBase() = default;
    NODISCARD virtual ParameterType type() const = 0;
    NODISCARD virtual std::string toString() const = 0;
    virtual void SetValue(const std::any&) = 0;
};

template<typename T>
class ParameterImpl : public ParameterBase {
public:
    ParameterImpl() : type_(GetParameterType<std::decay_t<T>>()) {}

    template<typename U,
             typename = typename std::enable_if_t<std::is_same_v<T, std::decay_t<U>>>>
    explicit ParameterImpl(U&& val)
        : type_(GetParameterType<std::decay_t<U>>()), value_(std::forward<U>(val)) {}

    NODISCARD ParameterType type() const override {
        return type_;
    }

    NODISCARD std::string toString() const override {
        if constexpr (GetParameterType<T>() == ParameterType::kParameterBool) {
            return value_ ? "True" : "False";
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterInt) {
            return std::to_string(value_);
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterFloat) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%e", value_);
            return buf;
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterString) {
            return value_;
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterComplex) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%e+%ei", value_.real(), value_.imag());
            return buf;
        }

        return "None";
    }

    void SetValue(const std::any& val) override {
        if constexpr (GetParameterType<T>() == ParameterType::kParameterBool) {
            value_ = std::any_cast<bool>(val);
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterInt) {

            value_ = std::any_cast<T>(val);
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterFloat) {
            value_ = std::any_cast<float>(val);
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterString) {
            value_ = std::any_cast<std::string>(val);
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterComplex) {
            value_ = std::any_cast<std::complex<float>>(val);
        }
    }

private:
    /**
     * @brief Parameter type
     */
    ParameterType type_{ParameterType::kParameterUnknown};

    /**
     * @brief Parameter value
     */
    T value_{};
};

class Parameter_ {
public:
    Parameter_() = default;

    template<typename T,
             typename = typename std::enable_if_t<!std::is_same_v<Parameter_, std::decay_t<T>>>>
    explicit Parameter_(T&& val)
        : ptr_(std::make_shared<ParameterImpl<std::decay_t<T>>>(std::forward<T>(val))) {}

    Parameter_(const Parameter_&) = default;

    Parameter_(Parameter_&& other) noexcept : ptr_(std::move(other.ptr_)) {}

    Parameter_& operator=(const Parameter_& other) {
        if (this != &other) {
            ptr_ = other.ptr_;
        }
        return *this;
    }

    Parameter_& operator=(Parameter_&& other) noexcept {
        if (this != &other) {
            ptr_ = std::move(other.ptr_);
        }
        return *this;
    }

    NODISCARD bool has_value() const {
        if (ptr_) {
            return true;
        }
        return false;
    }

    NODISCARD ParameterType type() const {
        if (has_value()) {
            return ptr_->type();
        }
        return ParameterType::kParameterUnknown;
    }

    template<typename T>
    std::optional<T> toValue() const {
        if (has_value()) {
            //
        }
        return {};
    }

    NODISCARD std::string toString() const {
        return ptr_->toString();
    }

private:
    std::shared_ptr<ParameterBase> ptr_;
};

template<typename T>
class Parameter {
public:
    using value_type = T;

    Parameter() : type_(GetParameterType<T>()) {}

    explicit Parameter(T val)
        : type_(GetParameterType<T>()),
          value_(val) {}

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


using ParameterVar = std::variant<Parameter<void*>,
                                  Parameter<bool>,
                                  Parameter<int>,
                                  Parameter<float>,
                                  Parameter<double>,
                                  Parameter<std::string>,
                                  Parameter<std::complex<float>>,

                                  Parameter<std::vector<int>>,
                                  Parameter<std::vector<float>>,
                                  Parameter<std::vector<std::string>>,
                                  Parameter<std::vector<std::complex<float>>>>;

}// namespace pnnx

#endif//OPENXAE_PARAMETER_H
