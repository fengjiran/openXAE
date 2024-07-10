//
// Created by richard on 6/14/24.
//

#ifndef OPENXAE_OPERAND_H
#define OPENXAE_OPERAND_H

#include "Parameter.h"
#include "utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace pnnx {

const int DimUnknownTag = -1;
const int DimVariableTag = -2333;

class Operator;
class Operand {
public:
    Operand() : type_(DataType::kDataTypeUnknown) {}

    explicit Operand(std::string name) : name_(std::move(name)), type_(DataType::kDataTypeUnknown) {}

    Operand(std::string name, DataType type, std::vector<int> shape)
        : name_(std::move(name)), type_(type), shape_(std::move(shape)) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (DimVariableTag == shape_[i]) {
                params_[std::string("__shape__") + std::to_string(i)] =
                        std::make_shared<Parameter>("arg" + std::to_string(i));
            }
        }
    }

    Operand(const Operand&) = delete;
    Operand(Operand&&) = delete;
    Operand& operator=(const Operand&) = delete;
    Operand& operator=(Operand&&) = delete;

    NODISCARD const DataType& type() const {
        return type_;
    }

    void SetType(DataType type) {
        type_ = type;
    }

    NODISCARD const std::vector<int>& GetShape() const {
        return shape_;
    }

    std::vector<int>& GetShape() {
        return shape_;
    }

    NODISCARD const std::string& name() const {
        return name_;
    }

    std::map<std::string, std::shared_ptr<Parameter>>& GetParams() {
        return params_;
    }

    NODISCARD const std::map<std::string, std::shared_ptr<Parameter>>& GetParams() const {
        return params_;
    }

    void SetProducer(const std::shared_ptr<Operator>& op) {
        producer_ = op;
    }

    void AddConsumer(const std::shared_ptr<Operator>& op) {
        consumers_.push_back(op);
    }

    NODISCARD const std::vector<std::shared_ptr<Operator>>& GetConsumers() const {
        return consumers_;
    }

    std::vector<std::shared_ptr<Operator>>& GetConsumers() {
        return consumers_;
    }

    void RemoveConsumer(const std::shared_ptr<Operator>& op) {
        auto it = std::find(consumers_.begin(), consumers_.end(), op);
        if (it != consumers_.end()) {
            consumers_.erase(it);
        }
    }

private:
    DataType type_;
    std::vector<int> shape_;
    std::shared_ptr<Operator> producer_;
    std::vector<std::shared_ptr<Operator>> consumers_;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string name_;
    std::map<std::string, std::shared_ptr<Parameter>> params_;
};

}// namespace pnnx

#endif//OPENXAE_OPERAND_H
