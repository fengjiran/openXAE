//
// Created by 赵丹 on 24-5-16.
//

#ifndef OPENXAE_UTILS_HPP
#define OPENXAE_UTILS_HPP

namespace XAcceleratorEngine {

template<typename T>
struct type_identity {
    using type = T;
};

template<typename T>
using type_identity_t = typename type_identity<T>::type;

template<typename T, T v>
struct integral_constant {
    static constexpr const T value = v;
    using value_type = T;
    using type = integral_constant;
    inline constexpr explicit operator value_type() const noexcept { return value; }
    inline constexpr value_type operator()() const noexcept { return value; }
};

}// namespace XAcceleratorEngine

#endif//OPENXAE_UTILS_HPP
