//
// Created by 赵丹 on 24-5-30.
//

#ifndef OPENXAE_IR_H
#define OPENXAE_IR_H

#include <climits>
#include <complex>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if BUILD_TORCH2PNNX
namespace torch {
namespace jit {
struct Value;
struct Node;
}// namespace jit
}// namespace torch
namespace at {
class Tensor;
}
#endif// BUILD_TORCH2PNNX

#if BUILD_ONNX2PNNX
namespace onnx {
class AttributeProto;
class TensorProto;
class ValueInfoProto;
}// namespace onnx
namespace pnnx {
namespace onnx2pnnx {
class OnnxAttributeProxy;
}// namespace onnx2pnnx
}// namespace pnnx
#endif// BUILD_ONNX2PNNX

namespace pnnx {

class Parameter {
public:
    Parameter() : type(0) {}
    explicit Parameter(bool b_) : type(1), b(b_) {}
    explicit Parameter(int i_) : type(2), i(i_) {}
    explicit Parameter(long l_) : type(2) {
        if (l_ == std::numeric_limits<long>::min()) {
            l_ = std::numeric_limits<int>::min();
        }

        if (l_ == std::numeric_limits<long>::max()) {
            l_ = std::numeric_limits<int>::max();
        }
        i = static_cast<int>(l_);
    }

    explicit Parameter(long long l_) : type(2) {
        if (l_ == std::numeric_limits<long long>::min()) {
            l_ = std::numeric_limits<int>::min();
        }

        if (l_ == std::numeric_limits<long long>::max()) {
            l_ = std::numeric_limits<int>::max();
        }
        i = static_cast<int>(l_);
    }

    explicit Parameter(float f_) : type(3), f(f_) {}
    explicit Parameter(double d_) : type(3), f((float) d_) {}
    explicit Parameter(const char* s_) : type(4), s(s_) {}
    explicit Parameter(std::string s_) : type(4), s(std::move(s_)) {}
    Parameter(const std::initializer_list<int>& ai_) : type(5), ai(ai_) {}
    Parameter(const std::initializer_list<int64_t>& ai_) : type(5) {
        for (const auto& x: ai_) {
            int64_t l_ = x;
            if (l_ == std::numeric_limits<long>::min()) {
                l_ = std::numeric_limits<int>::min();
            }

            if (l_ == std::numeric_limits<long>::max()) {
                l_ = std::numeric_limits<int>::max();
            }
            ai.push_back(static_cast<int>(l_));
        }
    }

    explicit Parameter(const std::vector<int>& ai_) : type(5), ai(ai_) {}
    explicit Parameter(const std::vector<int64_t>& ai_) : type(5) {
        for (const auto& x: ai_) {
            int64_t l_ = x;
            if (l_ == std::numeric_limits<long>::min()) {
                l_ = std::numeric_limits<int>::min();
            }

            if (l_ == std::numeric_limits<long>::max()) {
                l_ = std::numeric_limits<int>::max();
            }
            ai.push_back(static_cast<int>(l_));
        }
    }
    Parameter(const std::initializer_list<float>& af_) : type(6), af(af_) {}
    Parameter(const std::initializer_list<double>& af_) : type(6) {
        for (const auto& x: af_) {
            af.push_back((float) x);
        }
    }
    explicit Parameter(const std::vector<float>& af_) : type(6), af(af_) {}
    explicit Parameter(const std::vector<double>& af_) : type(6) {
        for (const auto& x: af_) {
            af.push_back((float) x);
        }
    }
    Parameter(const std::initializer_list<const char*>& as_) : type(7) {
        for (const auto& x: as_) {
            as.emplace_back(x);
        }
    }
    Parameter(const std::initializer_list<std::string>& as_) : type(7), as(as_) {}
    explicit Parameter(const std::vector<std::string>& as_) : type(7), as(as_) {}
    explicit Parameter(const std::complex<float>& c_) : type(10), c(c_) {}
    explicit Parameter(const std::complex<double>& c_) : type(10), c(c_) {}
    Parameter(const std::initializer_list<std::complex<float>>& ac_) : type(11), ac(ac_) {}
    Parameter(const std::initializer_list<std::complex<double>>& ac_) : type(11) {
        for (const auto& x: ac_) {
            ac.emplace_back(x);
        }
    }
    explicit Parameter(const std::vector<std::complex<float>>& ac_) : type(11), ac(ac_) {}
    explicit Parameter(const std::vector<std::complex<double>>& ac_) : type(11) {
        for (const auto& x: ac_) {
            ac.emplace_back(x);
        }
    }

#if BUILD_TORCH2PNNX
    Parameter(const torch::jit::Node* value_node);
    Parameter(const torch::jit::Value* value);
#endif// BUILD_TORCH2PNNX
#if BUILD_ONNX2PNNX
    Parameter(const onnx::AttributeProto& attr);
    Parameter(const onnx2pnnx::OnnxAttributeProxy& attr);
#endif// BUILD_ONNX2PNNX

    static Parameter parse_from_string(const std::string& value);
    static std::string encode_to_string(const Parameter& param);
    /**
     * @brief Parameter type
     *
     * 0 = null \n
     * 1 = bool \n
     * 2 = int \n
     * 3 = float \n
     * 4 = string \n
     * 5 = array int \n
     * 6 = array float \n
     * 7 = array string \n
     * 8 = others \n
     * 10 = complex \n
     * 11 = array complex
     */
    int type;

    bool b;
    int i;
    float f;
    std::complex<float> c;
    std::vector<int> ai;
    std::vector<float> af;
    std::vector<std::complex<float>> ac;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string s;
    std::vector<std::string> as;
};
bool operator==(const Parameter& lhs, const Parameter& rhs);

}// namespace pnnx


#endif//OPENXAE_IR_H
