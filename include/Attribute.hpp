//
// Created by 赵丹 on 24-5-14.
//

#ifndef OPENXAE_ATTRIBUTE_HPP
#define OPENXAE_ATTRIBUTE_HPP

#include "Datatype.hpp"
#include "glog/logging.h"
#include <memory>
#include <vector>

namespace XAcceleratorEngine {

/**
 * @brief Runtime operator attribute.
 *
 * Represents an attribute like weights or biases for a graph operator.
 * Contains the attribute data, shape, and data type.
 */
class Attribute {
public:
    Attribute() = default;

    Attribute(std::vector<uint32_t> shape_, DataType type_, std::vector<char> weight_)
        : shape(std::move(shape_)), type(type_), weight(std::move(weight_)) {}

    /**
     * @brief Attribute data.
     *
     * Typically contains the binary weight values.
     */
    std::vector<char> weight;

    /**
     * @brief Attribute data shape.
     *
     * Describes the dimensions of the attribute data.
     */
    std::vector<uint32_t> shape;

    /**
     * @brief Data type of the attribute.
     *
     * Such as float32, int8, etc.
     */
    DataType type = DataType::kTypeUnknown;

    /**
     * @brief Gets the attribute data as a typed array.
     * The attribute data is cleared after get by default.
     *
     * @tparam T Datatype to return (float, int, etc)
     * @param clearWeight Whether to clear data after get
     * @return Vector containing the attribute data
     */
    template<typename T>
    std::vector<T> get(bool clearWeight = true);
};

template<typename T>
std::vector<T> Attribute::get(bool clearWeight) {
    CHECK(!weight.empty());
    CHECK(type != DataType::kTypeUnknown);

    const auto elemSize = sizeof(T);
    CHECK(weight.size() % elemSize == 0);
    const auto elemCnt = weight.size() / elemSize;

    std::vector<T> tmp;
    tmp.reserve(elemCnt);
    switch (type) {
        case DataType::kTypeFloat32: {
            static_assert(std::is_same<T, float>::value);
            auto* weightPtr = reinterpret_cast<float*>(weight.data());
            for (size_t i = 0; i < elemCnt; ++i) {
                float elem = *(weightPtr + i);
                tmp.emplace_back(elem);
            }
            break;
        }
        default: {
            LOG(FATAL) << "Unknown weight data type: " << int32_t(type);
        }
    }

    if (clearWeight) {
        weight.clear();
    }

    return tmp;
}

}// namespace XAcceleratorEngine

#endif//OPENXAE_ATTRIBUTE_HPP
