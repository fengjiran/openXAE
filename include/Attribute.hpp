//
// Created by 赵丹 on 24-5-14.
//

#ifndef OPENXAE_ATTRIBUTE_HPP
#define OPENXAE_ATTRIBUTE_HPP

#include "Datatype.hpp"
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
};

}// namespace XAcceleratorEngine

#endif//OPENXAE_ATTRIBUTE_HPP
