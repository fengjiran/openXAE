//
// Created by 赵丹 on 24-5-13.
//

#ifndef OPENXAE_OPERAND_HPP
#define OPENXAE_OPERAND_HPP

#include "Tensor.hpp"

namespace XAcceleratorEngine {

/**
 * @brief Base for runtime graph operand
 *
 * Template base class representing an operand (input/output) in a
 * graph. Contains operand name, shape, data vector, and data type.
 *
 * @tparam T Operand data type (float, int, etc.)
 */
template<typename T>
class OperandBase {
public:
    OperandBase() = default;

    OperandBase(std::string name_, std::vector<uint32_t> shape_,
                std::vector<std::shared_ptr<Tensor<T>>> data_, DataType type_)
        : name(std::move(name_)), shape(std::move(shape_)), data(std::move(data_)), type(type_) {}

    size_t size() const;

    /// operand name
    std::string name;

    /// operand shape, e.g. NCHW
    std::vector<uint32_t> shape;

    /// shared_ptr vector of data
    std::vector<std::shared_ptr<Tensor<T>>> data;

    /// operand datatype
    DataType type = DataType::kTypeUnknown;
};

template<typename T>
size_t OperandBase<T>::size() const {
    if (shape.empty()) {
        return 0;
    }

    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    return size;
}

using Operand = OperandBase<float>;
using OperandQuantized = OperandBase<int8_t>;

}// namespace XAcceleratorEngine

#endif//OPENXAE_OPERAND_HPP
