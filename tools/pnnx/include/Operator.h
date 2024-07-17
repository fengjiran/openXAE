//
// Created by richard on 6/14/24.
//

#ifndef OPENXAE_OPERATOR_H
#define OPENXAE_OPERATOR_H

#include "Attribute.h"
#include "Operand.h"

namespace pnnx {

class Operator {
public:
    Operator() = default;

    Operator(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}

    Operator(std::string name, std::string type,
             std::map<std::string, std::shared_ptr<Parameter>> params,
             std::map<std::string, std::shared_ptr<Attribute>> attrs,
             std::vector<std::shared_ptr<Operand>> inputOperands,
             std::vector<std::string> inputNames)
        : name_(std::move(name)), type_(std::move(type)), params_(std::move(params)), attrs_(std::move(attrs)),
          inputOperands_(std::move(inputOperands)), inputNames_(std::move(inputNames)) {}

    Operator(const Operator&) = delete;
    Operator(Operator&&) = delete;
    Operator& operator=(const Operator&) = delete;
    Operator& operator=(Operator&&) = delete;

    NODISCARD bool HasParam(const std::string& key) const {
        return params_.find(key) != params_.end();
    }

    NODISCARD bool HasAttr(const std::string& key) const {
        return attrs_.find(key) != attrs_.end();
    }

    NODISCARD bool HasInput(const std::string& key) const {
        return std::find(inputNames_.begin(), inputNames_.end(), key) != inputNames_.end();
    }

    NODISCARD std::shared_ptr<Operand> GetNamedInput(const std::string& key) const {
        for (size_t i = 0; i < inputNames_.size(); ++i) {
            if (key == inputNames_[i]) {
                return inputOperands_[i];
            }
        }
        return {};
    }

    NODISCARD const std::string& type() const {
        return type_;
    }

    std::string& type() {
        return type_;
    }

    NODISCARD const std::string& name() const {
        return name_;
    }

    std::string& name() {
        return name_;
    }

    NODISCARD const std::vector<std::string>& GetInputNames() const {
        return inputNames_;
    }

    std::vector<std::string>& GetInputNames() {
        return inputNames_;
    }

    NODISCARD const std::vector<std::shared_ptr<Operand>>& GetInputOperands() const {
        return inputOperands_;
    }

    std::vector<std::shared_ptr<Operand>>& GetInputOperands() {
        return inputOperands_;
    }

    NODISCARD const std::vector<std::shared_ptr<Operand>>& GetOutputOperands() const {
        return outputOperands_;
    }

    std::vector<std::shared_ptr<Operand>>& GetOutputOperands() {
        return outputOperands_;
    }

    void AddInputOperand(const std::shared_ptr<Operand>& operand) {
        inputOperands_.push_back(operand);
    }

    void AddOutputOperand(const std::shared_ptr<Operand>& operand) {
        outputOperands_.push_back(operand);
    }

    NODISCARD const std::map<std::string, std::shared_ptr<Attribute>>& GetAttributes() const {
        return attrs_;
    }

    std::map<std::string, std::shared_ptr<Parameter>>& GetParameters() {
        return params_;
    }

    NODISCARD const std::map<std::string, std::shared_ptr<Parameter>>& GetParameters() const {
        return params_;
    }

    std::map<std::string, std::shared_ptr<Attribute>>& GetAttributes() {
        return attrs_;
    }

private:
    // keep std::string typed member the last for cross cxxabi compatibility
    std::string type_;
    std::string name_;

    std::vector<std::string> inputNames_;

    std::vector<std::shared_ptr<Operand>> inputOperands_;
    std::vector<std::shared_ptr<Operand>> outputOperands_;

    std::map<std::string, std::shared_ptr<Attribute>> attrs_;
    std::map<std::string, std::shared_ptr<Parameter>> params_;
};

}// namespace pnnx

#endif//OPENXAE_OPERATOR_H
