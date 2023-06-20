//
// Created by richard on 6/20/23.
//

/*!
 * \file src/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include "ir/expr.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include "../support/scalars.h"

namespace tvm {
    IntImm::IntImm(DataType dtype, int64_t value, Span span) {
        ICHECK(dtype.is_scalar()) << "ValueError: IntImm can only take scalar, but " << dtype
                                  << " was supplied.";
        ICHECK(dtype.is_int() || dtype.is_uint())
                << "ValueError: IntImm supports only int or uint type, but " << dtype << " was supplied.";
        if (dtype.is_uint()) {
            ICHECK_GE(value, 0U) << "ValueError: Literal value " << value
                                 << " is negative for unsigned integer type " << dtype;
            if (dtype.bits() < 64) {
                ICHECK_LT(value, 1LL << dtype.bits())
                        << "ValueError: Literal value " << value << " exceeds maximum of " << dtype;
            }
        } else if (dtype.bits() == 1) {
            // int(1)
            ICHECK(value == 0 || value == 1) << "ValueError: " << value << " exceeds range of " << dtype;
        } else if (dtype.bits() < 64) {
            ICHECK_GE(value, -(1LL << (dtype.bits() - 1)))
                    << "ValueError: Literal value " << value << " exceeds minimum of " << dtype;
            ICHECK_LT(value, 1LL << (dtype.bits() - 1))
                    << "ValueError: Literal value " << value << " exceeds maximum of " << dtype;
        }
        ObjectPtr<IntImmNode> node = make_object<IntImmNode>();
        node->dtype = dtype;
        node->value = value;
        node->span = span;
        data_ = std::move(node);
    }
    TVM_REGISTER_GLOBAL("ir.IntImm").set_body_typed([](DataType dtype, int64_t value, Span span) {
        return IntImm(dtype, value, span);
    });
    TVM_REGISTER_NODE_TYPE(IntImmNode);
}// namespace tvm