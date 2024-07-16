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
        if ((inputs_size == 4 || inputs_size == 5) && op->GetInputOperands()[0]->GetProducer()->GetParameters().at("value")->toValue<int>() != 1)
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

    if (op->type() == "aten::__and__" || op->type() == "aten::__or__" || op->type() == "aten::__xor__" || op->type() == "aten::__lshift__" || op->type() == "aten::__rshift__") {
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

}// namespace pnnx
