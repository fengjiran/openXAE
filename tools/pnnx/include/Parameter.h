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

// Get array size int compile time
template<typename T, std::size_t N>
constexpr std::size_t GetArraySize(T (&)[N]) noexcept {
    return N;
}

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
    ParameterImpl();

    template<typename U,
             typename = typename std::enable_if_t<std::is_same_v<T, std::decay_t<U>>>>
    explicit ParameterImpl(U&& val);

    NODISCARD const T& toValue() const;

    T& toValue();

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
        Parameter tmp(other);
        swap(tmp, *this);
        return *this;
    }

    Parameter& operator=(Parameter&& other) noexcept {
        Parameter tmp(other);
        swap(tmp, *this);
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

    static Parameter CreateParameterFromString(const std::string& value);

    friend void swap(Parameter& a, Parameter& b) noexcept {
        std::swap(a.ptr_, b.ptr_);
    }

private:
    std::shared_ptr<ParameterBase> ptr_;
};

bool operator==(const Parameter& lhs, const Parameter& rhs);

}// namespace pnnx

#endif//OPENXAE_PARAMETER_H
