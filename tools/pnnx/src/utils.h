//
// Created by fengj on 2024/5/31.
//

#ifndef OPENXAE_UTILS_H
#define OPENXAE_UTILS_H

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

}// namespace pnnx
#endif//OPENXAE_UTILS_H
