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
    explicit Attribute(const at::Tensor& t);
#endif

#if BUILD_ONNX2PNNX
    Attribute(const onnx::TensorProto& t);
#endif

    Attribute(const Attribute& other) = default;

    Attribute(Attribute&& other) noexcept
        : type_(other.type_), data_(std::move(other.data_)),
          shape_(std::move(other.shape_)), params_(std::move(other.params_)) {}

    Attribute& operator=(const Attribute& other) {
        Attribute tmp(other);
        swap(tmp, *this);
        return *this;
    }

    Attribute& operator=(Attribute&& other) noexcept {
        Attribute tmp(other);
        swap(tmp, *this);
        return *this;
    }

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

    std::map<std::string, std::shared_ptr<Parameter>>& GetParameters() {
        return params_;
    }

    NODISCARD const std::map<std::string, std::shared_ptr<Parameter>>& GetParameters() const {
        return params_;
    }

    // convenient routines for manipulate fp16/fp32 weight
    NODISCARD std::vector<float> CastToFloat32() const;

    void SetFloat32Data(const std::vector<float>& newData);

    friend void swap(Attribute& a, Attribute& b) noexcept {
        std::swap(a.type_, b.type_);
        std::swap(a.shape_, b.shape_);
        std::swap(a.data_, b.data_);
        std::swap(a.params_, b.params_);
    }

private:
    DataType type_;
    std::vector<int> shape_;
    std::vector<char> data_;
    std::map<std::string, std::shared_ptr<Parameter>> params_;
};

bool operator==(const Attribute& lhs, const Attribute& rhs);

/**
 * @brief Concat two attributes along the first axis.
 * @param a left attribute
 * @param b right attribute
 * @return new attribute object.
 */
Attribute operator+(const Attribute& a, const Attribute& b);

}// namespace pnnx

#endif//OPENXAE_ATTRIBUTE_H
