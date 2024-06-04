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

/**
 * @brief Runtime parameter type.
 *
 * Enumerates the parameter type supported for workload.
 */
enum class ParameterType {
    kParameterUnknown = 0,
    kParameterBool = 1,
    kParameterInt = 2,
    kParameterFloat = 3,
    kParameterString = 4,
    kParameterArrayInt = 5,
    kParameterArrayFloat = 6,
    kParameterArrayString = 7,
    kParameterComplex = 10,
    kParameterArrayComplex = 11
};

/**
 * @brief Runtime data type.
 */
enum class DataType {
    kDataTypeUnknown = 0,
    kDataTypeFloat32 = 1,
    kDataTypeFloat64 = 2,
    kDataTypeFloat16 = 3,
    kDataTypeInt32 = 4,
    kDataTypeInt64 = 5,
    kDataTypeInt16 = 6,
    kDataTypeInt8 = 7,
    kDataTypeUInt8 = 8,
    kDataTypeBool = 9,
    kDataTypeComplex64 = 10,
    kDataTypeComplex128 = 11,
    kDataTypeComplex32 = 12,
    kDataTypeBFloat16 = 13
};



class Parameter {
public:
    /**
     * @brief Default constructor.
     */
    Parameter() : type(ParameterType::kParameterUnknown) {}

    /**
     * @brief Constructor for bool type parameter.
     * @param b_val bool type value.
     */
    explicit Parameter(bool b_val) : type(ParameterType::kParameterBool), b(b_val) {}

    /**
     * @brief Constructor for int type parameter.
     * @param i_val int type value.
     */
    explicit Parameter(int i_val) : type(ParameterType::kParameterInt), i(i_val) {}

    /**
     * @brief Constructor for long type parameter.
     * @param l_val long type value.
     */
    explicit Parameter(long l_val) : type(ParameterType::kParameterInt) {
        if (l_val == std::numeric_limits<long>::min()) {
            l_val = std::numeric_limits<int>::min();
        }

        if (l_val == std::numeric_limits<long>::max()) {
            l_val = std::numeric_limits<int>::max();
        }
        i = static_cast<int>(l_val);
    }

    /**
     * @brief Constructor for long long type parameter.
     * @param ll_val long long type value.
     */
    explicit Parameter(long long ll_val) : type(ParameterType::kParameterInt) {
        if (ll_val == std::numeric_limits<long long>::min()) {
            ll_val = std::numeric_limits<int>::min();
        }

        if (ll_val == std::numeric_limits<long long>::max()) {
            ll_val = std::numeric_limits<int>::max();
        }
        i = static_cast<int>(ll_val);
    }

    /**
     * @brief Constructor for float type parameter.
     * @param f_val float type value.
     */
    explicit Parameter(float f_val) : type(ParameterType::kParameterFloat), f(f_val) {}

    /**
     * @brief Constructor for double type parameter.
     * @param d_val double type value.
     */
    explicit Parameter(double d_val) : type(ParameterType::kParameterFloat), f(static_cast<float>(d_val)) {}

    /**
     * @brief Constructor for string type parameter.
     * @param s_val string type value.
     */
    explicit Parameter(const char* s_val) : type(ParameterType::kParameterString), s(s_val) {}

    /**
     * @brief Constructor for string type parameter.
     * @param s_val string type value.
     */
    explicit Parameter(std::string s_val) : type(ParameterType::kParameterString), s(std::move(s_val)) {}

    /**
     * @brief Constructor for array int type parameter.
     * @param ai_val init list of int type value.
     */
    Parameter(const std::initializer_list<int>& ai_val) : type(ParameterType::kParameterArrayInt), ai(ai_val) {}

    /**
     * @brief Constructor for array int type parameter.
     * @param ai_val init list of int64 type value.
     */
    Parameter(const std::initializer_list<int64_t>& ai_val) : type(ParameterType::kParameterArrayInt) {
        for (const auto& x: ai_val) {
            int64_t l = x;
            if (l == std::numeric_limits<long>::min()) {
                l = std::numeric_limits<int>::min();
            }

            if (l == std::numeric_limits<long>::max()) {
                l = std::numeric_limits<int>::max();
            }
            ai.push_back(static_cast<int>(l));
        }
    }

    /**
     * @brief Constructor for array int type parameter.
     * @param ai_val vector of int type value.
     */
    explicit Parameter(const std::vector<int>& ai_val) : type(ParameterType::kParameterArrayInt), ai(ai_val) {}

    /**
     * @brief Constructor for array int64 type parameter.
     * @param ai_val vector of int64 type value.
     */
    explicit Parameter(const std::vector<int64_t>& ai_val) : type(ParameterType::kParameterArrayInt) {
        for (const auto& x: ai_val) {
            int64_t l = x;
            if (l == std::numeric_limits<long>::min()) {
                l = std::numeric_limits<int>::min();
            }

            if (l == std::numeric_limits<long>::max()) {
                l = std::numeric_limits<int>::max();
            }
            ai.push_back(static_cast<int>(l));
        }
    }

    /**
     * @brief Constructor for array float type parameter.
     * @param af_val init list of float type value.
     */
    Parameter(const std::initializer_list<float>& af_val) : type(ParameterType::kParameterArrayFloat), af(af_val) {}

    /**
     * @brief Constructor for array double type parameter.
     * @param af_val init list of double type value.
     */
    Parameter(const std::initializer_list<double>& af_val) : type(ParameterType::kParameterArrayFloat) {
        for (const auto& x: af_val) {
            af.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array float type parameter.
     * @param af_val vector of float type value.
     */
    explicit Parameter(const std::vector<float>& af_val) : type(ParameterType::kParameterArrayFloat), af(af_val) {}

    /**
     * @brief Constructor for array float type parameter.
     * @param af_val vector of double type value.
     */
    explicit Parameter(const std::vector<double>& af_val) : type(ParameterType::kParameterArrayFloat) {
        for (const auto& x: af_val) {
            af.push_back(static_cast<float>(x));
        }
    }

    /**
     * @brief Constructor for array string type parameter.
     * @param as_val init list of string type value.
     */
    Parameter(const std::initializer_list<const char*>& as_val) : type(ParameterType::kParameterArrayString) {
        for (const auto& x: as_val) {
            as.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array string type parameter.
     * @param as_val init list of string type value.
     */
    Parameter(const std::initializer_list<std::string>& as_val) : type(ParameterType::kParameterArrayString), as(as_val) {}

    /**
     * @brief Constructor for array string type parameter.
     * @param as_val vector of string type value.
     */
    explicit Parameter(const std::vector<std::string>& as_val) : type(ParameterType::kParameterArrayString), as(as_val) {}

    /**
     * @brief Constructor for complex type parameter.
     * @param c_val complex type value.
     */
    explicit Parameter(const std::complex<float>& c_val) : type(ParameterType::kParameterComplex), c(c_val) {}

    /**
     * @brief Constructor for complex type parameter.
     * @param c_val complex type value.
     */
    explicit Parameter(const std::complex<double>& c_val) : type(ParameterType::kParameterComplex), c(c_val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param ac_val init list of complex type value.
     */
    Parameter(const std::initializer_list<std::complex<float>>& ac_val) : type(ParameterType::kParameterArrayComplex), ac(ac_val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param ac_val init list of complex type value.
     */
    Parameter(const std::initializer_list<std::complex<double>>& ac_val) : type(ParameterType::kParameterArrayComplex) {
        for (const auto& x: ac_val) {
            ac.emplace_back(x);
        }
    }

    /**
     * @brief Constructor for array complex type parameter.
     * @param ac_val vector of complex type value.
     */
    explicit Parameter(const std::vector<std::complex<float>>& ac_val) : type(ParameterType::kParameterArrayComplex), ac(ac_val) {}

    /**
     * @brief Constructor for array complex type parameter.
     * @param ac_val vector of complex type value.
     */
    explicit Parameter(const std::vector<std::complex<double>>& ac_val) : type(ParameterType::kParameterArrayComplex) {
        for (const auto& x: ac_val) {
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

    static std::string encode_to_string(const Parameter& param);
    static Parameter parse_from_string(const std::string& value);

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
    ParameterType type;

    bool b{};
    int i{};
    float f{};
    std::complex<float> c;
    std::vector<int> ai;
    std::vector<float> af;
    std::vector<std::complex<float>> ac;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string s;
    std::vector<std::string> as;
};

bool operator==(const Parameter& lhs, const Parameter& rhs);

class Attribute {
public:
    /**
     * @brief Default constructor.
     */
    Attribute() : type(DataType::kDataTypeUnknown) {}

    Attribute(const std::initializer_list<int>& shape_, const std::vector<float>& t);

#if BUILD_TORCH2PNNX
    Attribute(const at::Tensor& t);
#endif
#if BUILD_ONNX2PNNX
    Attribute(const onnx::TensorProto& t);
#endif

    size_t elemsize() const;

    int elemcount() const;

    // convenient routines for manipulate fp16/fp32 weight
    std::vector<float> get_float32_data() const;
    void set_float32_data(const std::vector<float>& data_);
    /**
     * @brief Runtime attribute type.
     *
     * 0 = null \n
     * 1 = float32 \n
     * 2 = float64 \n
     * 3 = float16 \n
     * 4 = int32 \n
     * 5 = int64 \n
     * 6 = int16 \n
     * 7 = int8 \n
     * 8 = uint8 \n
     * 9 = bool \n
     * 10 = complex64 \n
     * 11 = complex128 \n
     * 12 = complex32 \n
     * 13 = bf16
     */
    DataType type;
    std::vector<int> shape;
    std::vector<char> data;
    std::map<std::string, Parameter> params;
};

bool operator==(const Attribute& lhs, const Attribute& rhs);

/**
 * @brief Concat two attributes along the first axis.
 * @param a left attribute
 * @param b right attribute
 * @return new attribute object.
 */
Attribute operator+(const Attribute& a, const Attribute& b);

class Operator;
class Operand {

};

}// namespace pnnx


#endif//OPENXAE_IR_H
