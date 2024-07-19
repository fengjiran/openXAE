//
// Created by fengj on 2024/5/31.
//

#ifndef OPENXAE_UTILS_H
#define OPENXAE_UTILS_H

#include "datatype.h"
#include <string>
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
}// namespace vision
#endif

namespace pnnx {

#define CONCAT_STR_(A, B) A##B
#define CONCAT_STR(A, B) CONCAT_STR_(A, B)

#if BUILD_TORCH2PNNX
const torch::jit::Node* FindNodeByKind(const std::shared_ptr<torch::jit::Graph>& graph, const std::string& kind);
#endif

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

bool IsInteger(DataType type);

std::string DataType2String(DataType type);

const char* DataType2NumpyString(DataType type);

const char* DataType2TorchString(DataType type);

size_t SizeOf(DataType type);

DataType String2Type(const std::string& s);

std::string GetBasename(const std::string& path);

void ParseStringList(char* s, std::vector<std::string>& list);

void PrintStringList(const std::vector<std::string>& list);

void ParseShapeList(char* s, std::vector<std::vector<int64_t>>& shapes, std::vector<std::string>& types);

void PrintShapeList(const std::vector<std::vector<int64_t>>& shapes, const std::vector<std::string>& types);

bool ModelFileMaybeTorchscript(const std::string& path);
}// namespace pnnx
#endif//OPENXAE_UTILS_H
