//
// Created by 赵丹 on 24-5-16.
//

#ifndef OPENXAE_UTILS_HPP
#define OPENXAE_UTILS_HPP

#include <iostream>

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

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template<typename alloc, typename = void>
struct _has_select_on_container_copy_construction : false_type {};

template<typename alloc>
struct _has_select_on_container_copy_construction<
        alloc,
        decltype((void) std::declval<alloc>().select_on_container_copy_construction())> : true_type {};

template<typename alloc,
         typename = typename std::enable_if<_has_select_on_container_copy_construction<const alloc>::value>::type>
inline constexpr static alloc select_on_container_copy_construction(const alloc& a) {
    return a.select_on_container_copy_construction();
}

template<typename alloc,
         typename = void,
         typename = typename std::enable_if<!_has_select_on_container_copy_construction<const alloc>::value>::type>
inline constexpr static alloc select_on_container_copy_construction(const alloc& a) {
    return a;
}

}// namespace XAcceleratorEngine

#endif//OPENXAE_UTILS_HPP
