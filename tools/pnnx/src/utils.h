//
// Created by fengj on 2024/5/31.
//

#ifndef OPENXAE_UTILS_H
#define OPENXAE_UTILS_H

#include <algorithm>
#include <climits>
#include <complex>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

//#ifdef __has_cpp_attribute
//#if __has_cpp_attribute(nodiscard) > 201907L
//#define NODISCARD [[nodiscard]]
//#else
//#define NODISCARD
//#endif
//#endif

#define NODISCARD [[nodiscard]]

#if BUILD_TORCH2PNNX
#include <memory>
namespace torch {
namespace jit {
struct Graph;
struct Node;
}// namespace jit
}// namespace torch
#endif

namespace pnnx {

#if BUILD_TORCH2PNNX
const torch::jit::Node* find_node_by_kind(const std::shared_ptr<torch::jit::Graph>& graph, const std::string& kind);
#endif

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

template<typename T>
constexpr bool is_string_v = std::is_same_v<std::decay_t<T>, std::string> || std::is_convertible_v<T, std::string>;

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

}// namespace pnnx
#endif//OPENXAE_UTILS_H
