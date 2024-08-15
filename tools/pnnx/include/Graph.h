//
// Created by richard on 6/14/24.
//

#ifndef OPENXAE_GRAPH_H
#define OPENXAE_GRAPH_H

#include "Parameter.h"
#include "Operator.h"

namespace pnnx {

class Graph {
public:
    /**
     * @brief Default constructor.
     */
    Graph() = default;

    Graph(const Graph&) = delete;

    Graph(Graph&&) = delete;

    Graph& operator=(const Graph&) = delete;

    Graph& operator=(Graph&&) = delete;

    int load(const std::string& paramPath, const std::string& binPath);

    int save(const std::string& paramPath, const std::string& binPath);

    int python(const std::string& pyPath, const std::string& binPath);

    int parse(const std::string& param);

    std::shared_ptr<Operator> CreateOperator(const std::string& type, const std::string& name);

    std::shared_ptr<Operand> CreateOperator(const std::string& type, const std::string& name,
                                            const std::map<std::string, std::shared_ptr<Parameter>>& params,
                                            const std::map<std::string, std::shared_ptr<Attribute>>& attrs,
                                            const std::vector<std::shared_ptr<Operand>>& inputOperands,
                                            const std::vector<std::string>& inputOperandNames,
                                            const std::string& outputName,
                                            DataType outputType,
                                            const std::vector<int>& outputShape);

    std::shared_ptr<Operator> CreateOperatorBefore(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur);

    std::shared_ptr<Operator> CreateOperatorAfter(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur);

    std::shared_ptr<Operand> CreateOperand(const std::string& name);

    std::shared_ptr<Operand> GetOperand(const std::string& name);

    const std::vector<std::shared_ptr<Operator>>& GetOperators() const {
        return ops_;
    }

    std::vector<std::shared_ptr<Operator>>& GetOperators() {
        return ops_;
    }

    const std::vector<std::shared_ptr<Operand>>& GetOperands() const {
        return operands_;
    }

    std::vector<std::shared_ptr<Operand>>& GetOperands() {
        return operands_;
    }

#if BUILD_TORCH2PNNX
    std::shared_ptr<Operand> CreateOperand(const torch::jit::Value* v);
#endif

#if BUILD_ONNX2PNNX
    Operand* new_operand(const onnx::ValueInfoProto& value);
    Operand* new_operand(const onnx::TensorProto& t);
#endif

private:
    std::vector<std::shared_ptr<Operator>> ops_;
    std::vector<std::shared_ptr<Operand>> operands_;
};

}

#endif//OPENXAE_GRAPH_H
