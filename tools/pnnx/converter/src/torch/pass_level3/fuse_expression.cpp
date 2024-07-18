//
// Created by richard on 7/16/24.
//

#include "fuse_expression.h"
#include "storezip.h"

#include <algorithm>

namespace pnnx {

static bool OperandMaybeShapeTensor(const std::shared_ptr<Operand>& operand) {
    const auto& op = operand->GetProducer();

    if (op->type() == "aten::size") {
        return op->GetInputOperands().size() == 1;
    }

    if (op->type() == "Tensor.to") {
        return OperandMaybeShapeTensor(op->GetInputOperands()[0]);
    }

    return false;
}

static bool OperandMaybeTensor(const std::shared_ptr<Operand>& operand) {
    const auto& op = operand->GetProducer();

    if (op->type() == "prim::Constant") {
        const std::shared_ptr<Parameter>& param = op->GetParameters().at("value");
        if (param->type() == ParameterType::kParameterUnknown ||
            param->type() == ParameterType::kParameterBool ||
            param->type() == ParameterType::kParameterInt ||
            param->type() == ParameterType::kParameterFloat ||
            param->type() == ParameterType::kParameterString ||
            param->type() == ParameterType::kParameterComplex) {
            return false;
        } else {
            return true;
        }
    }

    if (op->type() == "prim::NumToTensor") {
        return OperandMaybeTensor(op->GetInputOperands()[0]);
    }

    if (op->type() == "prim::ListConstruct") {
        return false;
    }

    if (op->type() == "torch.unbind" && op->GetInputOperands()[0]->GetShape().size() == 1) {
        return false;
    }

    if (op->type() == "aten::size") {
        return op->GetInputOperands().size() == 1;
    }

    if (op->type() == "Tensor.slice") {
        // static slice
        const size_t inputs_size = op->GetInputOperands().size();
        if (inputs_size != 3 && inputs_size != 4 && inputs_size != 5)
            return true;

        for (size_t i = 0; i < inputs_size; i++) {
            if (op->GetInputOperands()[i]->GetProducer()->type() != "prim::Constant")
                return true;

            if (op->GetInputOperands()[i]->GetProducer()->GetParameters().at("value")->type() !=
                ParameterType::kParameterInt)
                return true;
        }

        // dim=0
        if (inputs_size == 3 && op->GetParameters().at("dim")->toValue<int>() != 0)
            return true;
        if ((inputs_size == 4 || inputs_size == 5) &&
            op->GetInputOperands()[0]->GetProducer()->GetParameters().at("value")->toValue<int>() != 1)
            return true;

        // step=1
        if ((inputs_size == 3 || inputs_size == 4) && op->GetParameters().at("step")->toValue<int>() != 1)
            return true;
        if (inputs_size == 5 && op->GetInputOperands()[4]->GetProducer()->GetParameters().at("value")->toValue<int>() != 1)
            return true;

        return !OperandMaybeShapeTensor(op->GetInputOperands()[0]);
    }

    if (op->type() == "aten::Int") {
        return OperandMaybeTensor(op->GetInputOperands()[0]);
    }

    if (op->type() == "Tensor.to" || op->type() == "aten::detach") {
        return OperandMaybeTensor(op->GetInputOperands()[0]);
    }

    if (op->type() == "aten::ScalarImplicit") {
        return false;
    }

    if (op->type() == "aten::abs" ||
        op->type() == "aten::acos" ||
        op->type() == "aten::acosh" ||
        op->type() == "aten::asin" ||
        op->type() == "aten::asinh" ||
        op->type() == "aten::atan" ||
        op->type() == "aten::atanh" ||
        op->type() == "aten::ceil" ||
        op->type() == "aten::cos" ||
        op->type() == "aten::cosh" ||
        op->type() == "aten::exp" ||
        op->type() == "aten::floor" ||
        op->type() == "aten::log" ||
        op->type() == "aten::log10" ||
        op->type() == "aten::neg" ||
        op->type() == "aten::reciprocal" ||
        op->type() == "aten::round" ||
        op->type() == "aten::rsqrt" ||
        op->type() == "aten::sign" ||
        op->type() == "aten::sin" ||
        op->type() == "aten::sinh" ||
        op->type() == "aten::sqrt" ||
        op->type() == "aten::square" ||
        op->type() == "aten::tan" ||
        op->type() == "aten::tanh" ||
        op->type() == "aten::trunc") {
        return OperandMaybeTensor(op->GetInputOperands()[0]);
    }

    if (op->type() == "aten::atan2" ||
        op->type() == "aten::div" ||
        op->type() == "aten::floor_divide" ||
        op->type() == "aten::fmod" ||
        op->type() == "aten::max" ||
        op->type() == "aten::maximum" ||
        op->type() == "aten::min" ||
        op->type() == "aten::minimum" ||
        op->type() == "aten::mul" ||
        op->type() == "aten::pow" ||
        op->type() == "aten::remainder") {
        return OperandMaybeTensor(op->GetInputOperands()[0]) ||
               OperandMaybeTensor(op->GetInputOperands()[1]);
    }

    if (op->type() == "aten::__and__" ||
        op->type() == "aten::__or__" ||
        op->type() == "aten::__xor__" ||
        op->type() == "aten::__lshift__" ||
        op->type() == "aten::__rshift__") {
        return OperandMaybeTensor(op->GetInputOperands()[0]) ||
               OperandMaybeTensor(op->GetInputOperands()[1]);
    }

    if (op->type() == "aten::add" || op->type() == "aten::sub" || op->type() == "aten::rsub") {
        if (op->GetInputOperands().size() == 2)
            return OperandMaybeTensor(op->GetInputOperands()[0]) ||
                   OperandMaybeTensor(op->GetInputOperands()[1]);
        else// if (op->inputs.size() == 3)
            return OperandMaybeTensor(op->GetInputOperands()[0]) ||
                   OperandMaybeTensor(op->GetInputOperands()[1]) ||
                   OperandMaybeTensor(op->GetInputOperands()[2]);
    }

    return true;
}

static void fuse_expression(Graph& graph,
                            const std::shared_ptr<Operand>& operand,
                            std::string& expr,
                            std::vector<std::shared_ptr<Operand>>& inputs,
                            const std::set<std::string>& foldableConstants,
                            StoreZipReader& zip,
                            bool checkSubgraph = true) {
    //    Operator* op = operand->producer;
    const auto& op = operand->GetProducer();

    // fprintf(stderr, "fuse_expression %s %s\n", op->type.c_str(), operand->name.c_str());

    if (checkSubgraph && OperandMaybeTensor(operand)) {
        if (op->GetOutputOperands().size() > 1 || op->GetOutputOperands()[0]->GetConsumers().size() > 1) {
            goto DEFAULT;
        }
    }

    if (op->type() == "prim::Constant") {
        const std::shared_ptr<Parameter>& param = op->GetParameters()["value"];
        //         fprintf(stderr, "fuse_expression prim::Constant %d\n", param.type);
        if (param->type() == ParameterType::kParameterUnknown) {
            expr += "None";
        } else if (param->type() == ParameterType::kParameterBool) {
            expr += param->toValue<bool>() ? "True" : "False";
        } else if (param->type() == ParameterType::kParameterInt) {
            char tmp[32];
            sprintf(tmp, "%d", param->toValue<int>());
            expr += tmp;
        } else if (param->type() == ParameterType::kParameterFloat) {
            char tmp[32];
            sprintf(tmp, "%e", param->toValue<float>());
            expr += tmp;
        } else if (param->type() == ParameterType::kParameterString) {
            expr += param->toValue<std::string>();
        } else if (param->type() == ParameterType::kParameterArrayInt) {
            // ints
            expr += "[";
            for (int i = 0; i < (int) param->toValue<std::vector<int>>().size(); i++) {
                char tmp[32];
                sprintf(tmp, "%d", param->toValue<std::vector<int>>()[i]);
                expr += tmp;
                if (i != (int) param->toValue<std::vector<int>>().size() - 1)
                    expr += ",";
            }
            expr += "]";
        } else if (param->type() == ParameterType::kParameterArrayFloat) {
            // floats
            expr += "[";
            for (int i = 0; i < (int) param->toValue<std::vector<float>>().size(); i++) {
                char tmp[32];
                sprintf(tmp, "%e", param->toValue<std::vector<float>>()[i]);
                expr += tmp;
                if (i != (int) param->toValue<std::vector<float>>().size() - 1)
                    expr += ",";
            }
            expr += "]";
        } else if (param->type() == ParameterType::kParameterComplex) {
            char tmp[32];
            sprintf(tmp, "%e%+ej", param->toValue<std::complex<float>>().real(), param->toValue<std::complex<float>>().imag());
            expr += tmp;
        } else {
            goto DEFAULT;
        }
    } else if (op->type() == "pnnx.Attribute") {
        // fprintf(stderr, "operand pnnx.Attribute %s\n", operand->name.c_str());

        const std::shared_ptr<Attribute>& data = op->GetAttributes()["data"];
        if (data->GetShape().size() == 1 && data->GetShape()[0] == 1 && (int) data->type() != -1) {
            if (data->type() == DataType::kDataTypeUnknown) {
                expr += "None";
            } else if (data->type() == DataType::kDataTypeFloat32) {
                char tmp[32];
                sprintf(tmp, "%e", ((const float*) data->GetRawData().data())[0]);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeFloat64) {
                char tmp[32];
                sprintf(tmp, "%e", ((const double*) data->GetRawData().data())[0]);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeInt32) {
                char tmp[32];
                sprintf(tmp, "%d", ((const int*) data->GetRawData().data())[0]);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeInt64) {
                int64_t v = ((const int64_t*) data->GetRawData().data())[0];
                if (v == std::numeric_limits<int64_t>::max()) v = INT_MAX;
                if (v == std::numeric_limits<int64_t>::min()) v = INT_MIN;

                char tmp[32];
                sprintf(tmp, "%d", (int) v);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeInt16) {
                char tmp[32];
                sprintf(tmp, "%d", ((const short*) data->GetRawData().data())[0]);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeInt8) {
                char tmp[32];
                sprintf(tmp, "%d", ((const signed char*) data->GetRawData().data())[0]);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeUInt8) {
                char tmp[32];
                sprintf(tmp, "%u", ((const unsigned char*) data->GetRawData().data())[0]);
                expr += tmp;
            } else if (data->type() == DataType::kDataTypeBool) {
                expr += ((const char*) data->GetRawData().data())[0] ? "True" : "False";
            } else {
                // unsupported type
                std::cerr << "fuse expression got unsupported scalar type " << (int) data->type() << std::endl;
            }
        } else {
            goto DEFAULT;
        }
    } else if (op->type() == "torch.unbind") {
        // track chain
        // pnnx.Attribute/foldable with 1-rank
        // torch.unbind to constant scalar

        const std::shared_ptr<Operand>& operand2 = op->GetInputOperands()[0];
        if (operand2->GetProducer()->type() == "pnnx.Attribute") {
            const std::shared_ptr<Attribute>& data = operand2->GetProducer()->GetAttributes()["data"];

            if (data->GetShape().size() == 1 && (int) data->type() != -1) {
                // resolve scalar i
                int si = 0;
                for (size_t i = 0; i < op->GetOutputOperands().size(); i++) {
                    if (op->GetOutputOperands()[i] == operand) {
                        si = (int) i;
                        break;
                    }
                }

                if (data->type() == DataType::kDataTypeUnknown) {
                    expr += "None";
                } else if (data->type() == DataType::kDataTypeFloat32) {
                    char tmp[32];
                    sprintf(tmp, "%e", ((const float*) data->GetRawData().data())[si]);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeFloat64) {
                    char tmp[32];
                    sprintf(tmp, "%e", ((const double*) data->GetRawData().data())[si]);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeInt32) {
                    char tmp[32];
                    sprintf(tmp, "%d", ((const int*) data->GetRawData().data())[si]);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeInt64) {
                    int64_t v = ((const int64_t*) data->GetRawData().data())[si];
                    if (v == std::numeric_limits<int64_t>::max()) v = INT_MAX;
                    if (v == std::numeric_limits<int64_t>::min()) v = INT_MIN;

                    char tmp[32];
                    sprintf(tmp, "%d", (int) v);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeInt16) {
                    char tmp[32];
                    sprintf(tmp, "%d", ((const short*) data->GetRawData().data())[si]);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeInt8) {
                    char tmp[32];
                    sprintf(tmp, "%d", ((const signed char*) data->GetRawData().data())[si]);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeUInt8) {
                    char tmp[32];
                    sprintf(tmp, "%u", ((const unsigned char*) data->GetRawData().data())[si]);
                    expr += tmp;
                } else if (data->type() == DataType::kDataTypeBool) {
                    expr += ((const char*) data->GetRawData().data())[si] ? "True" : "False";
                } else {
                    // unsupported type
                    std::cerr << "fuse expression got unsupported scalar type %d\n"
                              << (int) data->type() << std::endl;
                    goto DEFAULT;
                }
                return;
            }
        }

        goto DEFAULT;
    } else if (checkSubgraph && OperandMaybeTensor(operand) &&
               foldableConstants.find(operand->name()) != foldableConstants.end()) {
        // fprintf(stderr, "operand_is_foldable %s\n", operand->name.c_str());

        if (operand->GetShape().empty() && (int) operand->type() != -1) {
            // fuse literal constant into expression
            if (operand->type() == DataType::kDataTypeUnknown) {
                expr += "None";
            } else if (operand->type() == DataType::kDataTypeFloat32) {
                float v;
                zip.read_file(operand->name(), (char*) &v);

                char tmp[32];
                sprintf(tmp, "%e", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeFloat64) {
                double v;
                zip.read_file(operand->name(), (char*) &v);

                char tmp[32];
                sprintf(tmp, "%e", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeInt32) {
                int v;
                zip.read_file(operand->name(), (char*) &v);

                char tmp[32];
                sprintf(tmp, "%d", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeInt64) {
                int64_t v;
                zip.read_file(operand->name(), (char*) &v);

                if (v == std::numeric_limits<int64_t>::max()) v = INT_MAX;
                if (v == std::numeric_limits<int64_t>::min()) v = INT_MIN;

                char tmp[32];
                sprintf(tmp, "%ld", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeInt16) {
                short v;
                zip.read_file(operand->name(), (char*) &v);

                char tmp[32];
                sprintf(tmp, "%d", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeInt8) {
                signed char v;
                zip.read_file(operand->name(), (char*) &v);

                char tmp[32];
                sprintf(tmp, "%d", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeUInt8) {
                unsigned char v;
                zip.read_file(operand->name(), (char*) &v);

                char tmp[32];
                sprintf(tmp, "%u", v);
                expr += tmp;
            } else if (operand->type() == DataType::kDataTypeBool) {
                char v;
                zip.read_file(operand->name(), &v);

                expr += v ? "True" : "False";
            } else {
                // fprintf(stderr, "unknown foldable literal %s %d\n", operand->name.c_str(), operand->type);
                auto it = std::find(inputs.begin(), inputs.end(), operand);
                if (it == inputs.end()) {
                    // tensor
                    char tmp[32];
                    sprintf(tmp, "@%d", (int) inputs.size());
                    expr += tmp;

                    inputs.push_back(operand);
                } else {
                    // tensor
                    char tmp[32];
                    sprintf(tmp, "@%d", (int) (it - inputs.begin()));
                    expr += tmp;
                }
            }
        } else {
            goto DEFAULT;
        }
    } else if (op->type() == "prim::NumToTensor") {
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
    } else if (op->type() == "prim::ListConstruct") {
        expr += "[";
        for (int i = 0; i < (int) op->GetInputOperands().size() - 1; i++) {
            fuse_expression(graph, op->GetInputOperands()[i], expr, inputs, foldableConstants, zip);
            expr += ",";
        }
        if (!op->GetInputOperands().empty()) {
            fuse_expression(graph, op->GetInputOperands()[op->GetInputOperands().size() - 1], expr, inputs, foldableConstants, zip);
        }
        expr += "]";
    } else if (op->type() == "aten::size") {
        if (op->GetInputOperands().size() == 1) {
            fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        } else// if (op->inputs.size() == 2)
        {
            expr += "size(";
            fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
            expr += ",";
            fuse_expression(graph, op->GetInputOperands()[1], expr, inputs, foldableConstants, zip);
            expr += ")";
        }
    } else if (op->type() == "Tensor.slice" && !OperandMaybeTensor(operand)) {
        int start = op->GetInputOperands().size() == 3
                            ? op->GetInputOperands()[1]->GetProducer()->GetParameters().at("value")->toValue<int>()
                            : op->GetInputOperands()[2]->GetProducer()->GetParameters().at("value")->toValue<int>();
        int end = op->GetInputOperands().size() == 3
                          ? op->GetInputOperands()[2]->GetProducer()->GetParameters().at("value")->toValue<int>()
                          : op->GetInputOperands()[3]->GetProducer()->GetParameters().at("value")->toValue<int>();

        // onnx style shape + slice chain
        const std::shared_ptr<Operator>& op_shape = op->GetInputOperands()[0]->GetProducer();
        for (int i = start; i < end; i++) {
            expr += "size(";
            fuse_expression(graph, op_shape->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
            expr += ",";
            expr += std::to_string(i);
            expr += ")";
            if (i + 1 != end)
                expr += ",";
        }
    } else if (op->type() == "aten::Int") {
        expr += "int(";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ")";
    } else if (op->type() == "Tensor.to") {
        bool noop_type_cast = ((int) op->GetOutputOperands()[0]->type() != -1) && (op->GetInputOperands()[0]->type() == op->GetOutputOperands()[0]->type());
        if (noop_type_cast) {
            fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        } else if (!OperandMaybeTensor(operand)) {
            std::string dtype = op->GetParameters().at("dtype")->toValue<std::string>();

            // torch.xxx
            expr += dtype + "(";
            fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
            expr += ")";
        } else {
            goto DEFAULT;
        }
    } else if (op->type() == "aten::detach" || op->type() == "aten::ScalarImplicit") {
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
    } else if (op->type() == "aten::abs" ||
               op->type() == "aten::acos" ||
               op->type() == "aten::acosh" ||
               op->type() == "aten::asin" ||
               op->type() == "aten::asinh" ||
               op->type() == "aten::atan" ||
               op->type() == "aten::atanh" ||
               op->type() == "aten::ceil" ||
               op->type() == "aten::cos" ||
               op->type() == "aten::cosh" ||
               op->type() == "aten::exp" ||
               op->type() == "aten::floor" ||
               op->type() == "aten::log" ||
               op->type() == "aten::log10" ||
               op->type() == "aten::neg" ||
               op->type() == "aten::reciprocal" ||
               op->type() == "aten::round" ||
               op->type() == "aten::rsqrt" ||
               op->type() == "aten::sign" ||
               op->type() == "aten::sin" ||
               op->type() == "aten::sinh" ||
               op->type() == "aten::sqrt" ||
               op->type() == "aten::square" ||
               op->type() == "aten::tan" ||
               op->type() == "aten::tanh" ||
               op->type() == "aten::trunc") {
        std::string mathop = op->type().substr(6);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ")";
    } else if (op->type() == "aten::atan2" ||
               op->type() == "aten::floor_divide" ||
               op->type() == "aten::fmod" ||
               op->type() == "aten::max" ||
               op->type() == "aten::maximum" ||
               op->type() == "aten::min" ||
               op->type() == "aten::minimum" ||
               op->type() == "aten::mul" ||
               op->type() == "aten::pow" ||
               op->type() == "aten::remainder") {
        std::string mathop = op->type().substr(6);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ",";
        fuse_expression(graph, op->GetInputOperands()[1], expr, inputs, foldableConstants, zip);
        expr += ")";
    } else if (op->type() == "aten::__and__" ||
               op->type() == "aten::__or__" ||
               op->type() == "aten::__xor__" ||
               op->type() == "aten::__lshift__" ||
               op->type() == "aten::__rshift__") {
        std::string mathop = op->type().substr(8, op->type().size() - 10);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ",";
        fuse_expression(graph, op->GetInputOperands()[1], expr, inputs, foldableConstants, zip);
        expr += ")";
    } else if (op->type() == "aten::add" || op->type() == "aten::sub") {
        std::string mathop = op->type().substr(6);

        expr += mathop;
        expr += "(";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ",";

        std::string expr1;
        fuse_expression(graph, op->GetInputOperands()[1], expr1, inputs, foldableConstants, zip);

        if (op->GetInputOperands().size() == 2) {
            expr += expr1;
        } else// if (op->inputs.size() == 3)
        {
            std::string expr2;
            fuse_expression(graph, op->GetInputOperands()[2], expr2, inputs, foldableConstants, zip);

            if (expr2 == "1") {
                expr += expr1;
            } else {
                expr += ",";
                expr += "mul(";
                expr += expr1;
                expr += ",";
                expr += expr2;
                expr += ")";
            }
        }

        expr += ")";
    } else if (op->type() == "aten::rsub") {
        expr += "sub(";
        std::string expr1;
        fuse_expression(graph, op->GetInputOperands()[1], expr1, inputs, foldableConstants, zip);

        if (op->GetInputOperands().size() == 2) {
            expr += expr1;
        } else// if (op->inputs.size() == 3)
        {
            std::string expr2;
            fuse_expression(graph, op->GetInputOperands()[2], expr2, inputs, foldableConstants, zip);

            if (expr2 == "1") {
                expr += expr1;
            } else {
                expr += ",";
                expr += "mul(";
                expr += expr1;
                expr += ",";
                expr += expr2;
                expr += ")";
            }
        }

        expr += ",";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ")";
    } else if (op->type() == "aten::div") {
        std::string rounding_mode;
        if (op->GetInputOperands().size() == 3)
            fuse_expression(graph, op->GetInputOperands()[2], rounding_mode, inputs, foldableConstants, zip);

        if (rounding_mode == "trunc") {
            expr += "floor_divide";
        } else {
            expr += "div";
        }

        expr += "(";
        fuse_expression(graph, op->GetInputOperands()[0], expr, inputs, foldableConstants, zip);
        expr += ",";
        fuse_expression(graph, op->GetInputOperands()[1], expr, inputs, foldableConstants, zip);
        expr += ")";
    } else {
        goto DEFAULT;
    }

    return;

DEFAULT:
    auto it = std::find(inputs.begin(), inputs.end(), operand);
    if (it == inputs.end()) {
        // tensor
        char tmp[32];
        sprintf(tmp, "@%d", (int) inputs.size());
        expr += tmp;

        inputs.push_back(operand);
    } else {
        // tensor
        char tmp[32];
        sprintf(tmp, "@%d", (int) (it - inputs.begin()));
        expr += tmp;
    }
}

void fuse_expression(Graph& graph,
                     const std::set<std::string>& foldableConstants,
                     const std::string& foldableConstantsZippath) {
    StoreZipReader zip;
    zip.open(foldableConstantsZippath);

    int pnnx_expr_index = 0;

    for (;;) {
        bool need_fuse = false;

        // build expression via reverse order
        for (int i = (int) graph.GetOperators().size() - 1; i >= 0; i--) {
            std::shared_ptr<Operator> op = graph.GetOperators()[i];

            if (op->type() == "prim::Constant") {
                need_fuse = true;
            }
            if (op->type() == "prim::NumToTensor") {
                need_fuse = true;
            }
            if (op->type() == "prim::ListConstruct") {
                need_fuse = true;
            }
            if (op->type() == "aten::size") {
                need_fuse = true;
            }
            if (op->type() == "aten::Int") {
                need_fuse = true;
            }
            if (op->type() == "Tensor.to") {
                // fuse noop type cast only
                bool noop_to = ((int) op->GetOutputOperands()[0]->type() != -1) &&
                               (op->GetInputOperands()[0]->type() == op->GetOutputOperands()[0]->type());
                bool is_scalar = !OperandMaybeTensor(op->GetOutputOperands()[0]);
                need_fuse = noop_to || is_scalar;
            }
            if (op->type() == "aten::detach" || op->type() == "aten::ScalarImplicit") {
                need_fuse = true;
            }
            if (op->type() == "aten::abs" ||
                op->type() == "aten::acos" ||
                op->type() == "aten::acosh" ||
                op->type() == "aten::add" ||
                op->type() == "aten::asin" ||
                op->type() == "aten::asinh" ||
                op->type() == "aten::atan" ||
                op->type() == "aten::atanh" ||
                op->type() == "aten::atan2" ||
                op->type() == "aten::ceil" ||
                op->type() == "aten::cos" ||
                op->type() == "aten::cosh" ||
                op->type() == "aten::div" ||
                op->type() == "aten::exp" ||
                op->type() == "aten::floor" ||
                op->type() == "aten::floor_divide" ||
                op->type() == "aten::fmod" ||
                op->type() == "aten::log" ||
                op->type() == "aten::log10" ||
                op->type() == "aten::max" ||
                op->type() == "aten::maximum" ||
                op->type() == "aten::min" ||
                op->type() == "aten::minimum" ||
                op->type() == "aten::mul" ||
                op->type() == "aten::neg" ||
                op->type() == "aten::pow" ||
                op->type() == "aten::reciprocal" ||
                op->type() == "aten::remainder" ||
                op->type() == "aten::round" ||
                op->type() == "aten::rsqrt" ||
                op->type() == "aten::rsub" ||
                op->type() == "aten::sign" ||
                op->type() == "aten::sin" ||
                op->type() == "aten::sinh" ||
                op->type() == "aten::sqrt" ||
                op->type() == "aten::square" ||
                op->type() == "aten::sub" ||
                op->type() == "aten::tan" ||
                op->type() == "aten::tanh" ||
                op->type() == "aten::trunc") {
                need_fuse = true;
            }
            if (op->type() == "aten::__and__" ||
                op->type() == "aten::__or__" ||
                op->type() == "aten::__xor__" ||
                op->type() == "aten::__lshift__" ||
                op->type() == "aten::__rshift__") {
                need_fuse = true;
            }

            if (need_fuse) {
                std::string expr;
                std::vector<std::shared_ptr<Operand>> inputs;
                fuse_expression(graph, op->GetOutputOperands()[0], expr, inputs, foldableConstants, zip, false);
                //                 fprintf(stderr, "expr = %s\n", expr.c_str());

                // lets rewrite graph
                char name[32];
                sprintf(name, "pnnx_expr_%d", pnnx_expr_index++);

                op->type() = "pnnx.Expression";
                op->name() = name;

                op->GetParameters().clear();
                op->GetAttributes().clear();

                op->GetParameters()["expr"] = std::make_shared<Parameter>(expr);

                // fix input output
                for (auto& operand: op->GetInputOperands()) {
                    operand->GetConsumers().erase(std::find(operand->GetConsumers().begin(), operand->GetConsumers().end(), op));
                }

                op->GetInputOperands() = inputs;

                for (auto& operand: op->GetInputOperands()) {
                    operand->GetConsumers().push_back(op);
                }

                break;
            }
        }

        if (!need_fuse)
            break;
    }

    zip.close();
}

}// namespace pnnx
