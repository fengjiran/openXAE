//
// Created by 赵丹 on 24-6-14.
//

#include "Attribute.h"

#include <numeric>
#include <cstring>
#include <iostream>

namespace pnnx {

Attribute::Attribute(const std::vector<int>& shape, const std::vector<float>& t)
    : type_(DataType::kDataTypeFloat32), shape_(shape) {
    if (!shape_.empty()) {
        data_.resize(size() * GetElemSize());
        memcpy((void*) data_.data(), (const void*) t.data(), data_.size());
    }
}

size_t Attribute::GetElemSize() const {
    return SizeOf(type_);
}

size_t Attribute::size() const {
    if (shape_.empty()) {
        return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
}

std::vector<float> Attribute::CastToFloat32() const {
    std::vector<float> v(size());
    if (type() == DataType::kDataTypeFloat32) {
        memcpy((void*) v.data(), (const void*) data_.data(), data_.size());
    } else if (type() == DataType::kDataTypeFloat64) {
        const auto* p = (const double*) data_.data();
        for (auto& item: v) {
            item = static_cast<float>(*p++);
        }
    } else if (type() == DataType::kDataTypeFloat16) {
        const auto* p = (const unsigned short*) data_.data();
        for (auto& item: v) {
            item = float16_to_float32(*p++);
        }
    } else {
        std::cerr << "Cannot convert to float32 type.\n";
    }
    return v;
}

void Attribute::SetFloat32Data(const std::vector<float>& newData) {
    data_.resize(newData.size() * GetElemSize());
    switch (type()) {
        case DataType::kDataTypeFloat32: {
            memcpy((void*) data_.data(), (const void*) newData.data(), data_.size());
            break;
        }

        case DataType::kDataTypeFloat64: {
            auto* p = (double*) data_.data();
            for (const auto& item: newData) {
                *p = item;
                ++p;
            }
            break;
        }

        case DataType::kDataTypeFloat16: {
            auto* p = (unsigned short*) data_.data();
            for (const auto& item: newData) {
                *p = float32_to_float16(item);
                ++p;
            }
        }

        default:
            std::cerr << "Cannot convert to float32 type.\n";
    }
}

bool operator==(const Attribute& lhs, const Attribute& rhs) {
    if (lhs.type() != rhs.type()) {
        return false;
    }

    if (lhs.type() == DataType::kDataTypeUnknown) {
        return true;
    }

    if (lhs.GetShape() != rhs.GetShape()) {
        return false;
    }

    if (lhs.GetRawData() != rhs.GetRawData()) {
        return false;
    }
    return true;
}

Attribute operator+(const Attribute& a, const Attribute& b) {
    Attribute c;
    if (a.type() != b.type()) {
        std::cerr << "concat attribute type mismatch\n";
        return c;
    }

    if (a.GetShape() != b.GetShape()) {
        std::cerr << "concat attribute shape mismatch\n";
        return c;
    }

    c.SetType(a.type());
    c.GetShape() = a.GetShape();
    c.GetShape()[0] += b.GetShape()[0];// concat the first dim

    c.GetRawData().resize(a.GetRawData().size() + b.GetRawData().size());
    memcpy(c.GetRawData().data(), a.GetRawData().data(), a.GetRawData().size());
    memcpy(c.GetRawData().data() + a.GetRawData().size(), b.GetRawData().data(), b.GetRawData().size());
    return c;
}

}