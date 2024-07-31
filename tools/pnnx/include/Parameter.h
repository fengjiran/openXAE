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
    NODISCARD virtual ParameterType& type() = 0;
    NODISCARD virtual const ParameterType& type() const = 0;
    NODISCARD virtual std::string toString() const = 0;

    virtual ~ParameterBase() = default;
};

template<typename T>
class ParameterImpl : public ParameterBase {
public:
    ParameterImpl() : type_(GetParameterType<std::decay_t<T>>()) {}

    template<typename U,
             typename = typename std::enable_if_t<std::is_same_v<T, std::decay_t<U>>>>
    explicit ParameterImpl(U&& val)
        : type_(GetParameterType<std::decay_t<U>>()), value_(std::forward<U>(val)) {}

    NODISCARD const T& toValue() const {
        return value_;
    }

    NODISCARD T& toValue() {
        return value_;
    }

    NODISCARD ParameterType& type() override {
        return type_;
    }

    NODISCARD const ParameterType& type() const override {
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
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterArrayInt) {
            std::string code;
            code += "(";
            size_t size = toValue().size();
            for (const auto& ele: toValue()) {
                code += (std::to_string(ele) + (--size ? "," : ""));
            }
            code += ")";
            return code;
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterArrayFloat) {
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
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterArrayString) {
            std::string code;
            code += "(";
            size_t size = toValue().size();
            for (const auto& ele: toValue()) {
                code += (ele + (--size ? "," : ""));
            }
            code += ")";
            return code;
        } else if constexpr (GetParameterType<T>() == ParameterType::kParameterArrayComplex) {
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

        return "None";
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

class Parameter {
public:
    Parameter() = default;

    template<typename T,
             typename = typename std::enable_if_t<!std::is_same_v<Parameter, std::decay_t<T>>>>
    explicit Parameter(T&& val) {
        SetValue(std::forward<T>(val));
    }

    template<typename ElemType>
    Parameter(std::initializer_list<ElemType> il) {
        SetValue(std::move(il));
    }

    Parameter(const Parameter&) = default;

    Parameter(Parameter&& other) noexcept : ptr_(std::move(other.ptr_)) {}

    Parameter& operator=(const Parameter& other) {
        if (this != &other) {
            ptr_ = other.ptr_;
        }
        return *this;
    }

    Parameter& operator=(Parameter&& other) noexcept {
        if (this != &other) {
            ptr_ = std::move(other.ptr_);
        }
        return *this;
    }

    template<typename T>
    Parameter& operator=(T&& val) {
        SetValue(std::forward<T>(val));
        return *this;
    }

    template<typename T>
    void SetValue(T&& val) {
        using U = std::decay_t<T>;
        if constexpr (std::is_integral_v<U> && !std::is_same_v<U, bool>) {
            ptr_ = std::make_shared<ParameterImpl<int>>((int) std::forward<T>(val));
        } else if constexpr (std::is_floating_point_v<U>) {
            ptr_ = std::make_shared<ParameterImpl<float>>((float) std::forward<T>(val));
        } else if constexpr (is_string_v<T>) {
            ptr_ = std::make_shared<ParameterImpl<std::string>>(std::string(std::forward<T>(val)));
        } else if constexpr (is_std_vector_int_v<U>) {
            ptr_ = std::make_shared<ParameterImpl<std::vector<int>>>(
                    std::vector<int>(std::forward<T>(val).begin(), std::forward<T>(val).end()));
        } else if constexpr (is_std_vector_float_v<U>) {
            ptr_ = std::make_shared<ParameterImpl<std::vector<float>>>(
                    std::vector<float>(std::forward<T>(val).begin(), std::forward<T>(val).end()));
        } else if constexpr (is_std_vector_string_v<U>) {
            ptr_ = std::make_shared<ParameterImpl<std::vector<std::string>>>(
                    std::vector<std::string>(std::forward<T>(val).begin(), std::forward<T>(val).end()));
        } else {
            ptr_ = std::make_shared<ParameterImpl<U>>(std::forward<T>(val));
        }
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

    void SetOtherType() {
        if (has_value()) {
            ptr_->type() = ParameterType::kParameterOther;
        }
    }

    template<typename T>
    const T& toValue() const {
        if (!has_value()) {
            throw std::bad_cast();
        }

        auto ptr = std::dynamic_pointer_cast<ParameterImpl<T>>(ptr_);
        if (ptr) {
            return ptr->toValue();
        } else {
            throw std::bad_cast();
        }
    }

    template<typename T>
    T& toValue() {
        if (!has_value()) {
            throw std::bad_cast();
        }

        auto ptr = std::dynamic_pointer_cast<ParameterImpl<T>>(ptr_);
        if (ptr) {
            return ptr->toValue();
        } else {
            throw std::bad_cast();
        }
    }

    NODISCARD std::string toString() const {
        if (has_value()) {
            return ptr_->toString();
        }
        return "None";
    }

private:
    std::shared_ptr<ParameterBase> ptr_;
};


template<typename T>
class Parameter_Deprecated {
public:
    using value_type = T;

    Parameter_Deprecated() : type_(GetParameterType<T>()) {}

    explicit Parameter_Deprecated(T val)
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
class Parameter_Deprecated<std::vector<T>> {
public:
    using value_type = std::vector<T>;

    Parameter_Deprecated() : type_(GetParameterType<std::vector<T>>()) {}

    explicit Parameter_Deprecated(const std::vector<T>& val)
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
Parameter_Deprecated(const char*) -> Parameter_Deprecated<std::string>;

Parameter_Deprecated(std::initializer_list<const char*>) -> Parameter_Deprecated<std::vector<std::string>>;

Parameter_Deprecated(std::vector<const char*>) -> Parameter_Deprecated<std::vector<std::string>>;

template<typename T>
Parameter_Deprecated(std::initializer_list<T>) -> Parameter_Deprecated<std::vector<T>>;

template<typename T>
bool operator==(const Parameter_Deprecated<T>& lhs, const Parameter_Deprecated<T>& rhs) {
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
    return Parameter_Deprecated<Args...>(std::forward<Args>(args)...);
}


using ParameterVar = std::variant<Parameter_Deprecated<void*>,
                                  Parameter_Deprecated<bool>,
                                  Parameter_Deprecated<int>,
                                  Parameter_Deprecated<float>,
                                  Parameter_Deprecated<double>,
                                  Parameter_Deprecated<std::string>,
                                  Parameter_Deprecated<std::complex<float>>,

                                  Parameter_Deprecated<std::vector<int>>,
                                  Parameter_Deprecated<std::vector<float>>,
                                  Parameter_Deprecated<std::vector<std::string>>,
                                  Parameter_Deprecated<std::vector<std::complex<float>>>>;

}// namespace pnnx

#endif//OPENXAE_PARAMETER_H
