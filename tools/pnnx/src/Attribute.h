//
// Created by 赵丹 on 24-6-14.
//

#ifndef OPENXAE_ATTRIBUTE_H
#define OPENXAE_ATTRIBUTE_H

#include "utils.h"
#include <vector>

namespace pnnx {

class Attribute {
public:
    /**
     * @brief Default constructor.
     */
    Attribute() : type_(DataType::kDataTypeUnknown) {}

    Attribute(const std::vector<int>& shape, const std::vector<float>& t);

#if BUILD_TORCH2PNNX
    Attribute(const at::Tensor& t);
#endif
#if BUILD_ONNX2PNNX
    Attribute(const onnx::TensorProto& t);
#endif

    Attribute(const Attribute&) = delete;
    Attribute(Attribute&& other) noexcept
        : type_(other.type_), data_(std::move(other.data_)), shape_(std::move(other.shape_)) {}

    Attribute& operator=(const Attribute&) = delete;
    Attribute& operator=(Attribute&&) = delete;

    NODISCARD const DataType& type() const {
        return type_;
    }

    void SetType(DataType type) {
        type_ = type;
    }

    NODISCARD size_t GetElemSize() const;

    NODISCARD size_t size() const;

    NODISCARD const std::vector<int>& GetShape() const {
        return shape_;
    }

    std::vector<int>& GetShape() {
        return shape_;
    }

    NODISCARD const std::vector<char>& GetRawData() const {
        return data_;
    }

    std::vector<char>& GetRawData() {
        return data_;
    }

    // convenient routines for manipulate fp16/fp32 weight
    NODISCARD std::vector<float> CastToFloat32() const;

    void SetFloat32Data(const std::vector<float>& newData);

private:

    DataType type_;
    std::vector<int> shape_;
    std::vector<char> data_;
    //    std::map<std::string, Parameter> params;
};

bool operator==(const Attribute& lhs, const Attribute& rhs);

/**
 * @brief Concat two attributes along the first axis.
 * @param a left attribute
 * @param b right attribute
 * @return new attribute object.
 */
Attribute operator+(const Attribute& a, const Attribute& b);

}

#endif//OPENXAE_ATTRIBUTE_H
