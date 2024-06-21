//
// Created by fengj on 2024/5/31.
//

#ifndef OPENXAE_UTILS_H
#define OPENXAE_UTILS_H

#include "datatype.h"
#include <string>

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
struct Value;
}// namespace jit
}// namespace torch

namespace at {
class Tensor;
}
#endif

#ifdef PNNX_TORCHVISION
namespace vision {
int64_t cuda_version();
} // namespace vision
#endif

namespace pnnx {

#if BUILD_TORCH2PNNX
const torch::jit::Node* find_node_by_kind(const std::shared_ptr<torch::jit::Graph>& graph, const std::string& kind);
#endif

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

bool IsInteger(DataType type);

std::string DataType2String(DataType type);

const char* DataType2NumpyString(DataType type);

const char* DataType2TorchString(DataType type);

size_t SizeOf(DataType type);

DataType String2Type(const std::string& s);


}// namespace pnnx
#endif//OPENXAE_UTILS_H
