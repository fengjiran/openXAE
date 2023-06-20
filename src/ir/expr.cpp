//
// Created by richard on 6/20/23.
//

/*!
 * \file src/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include "ir/expr.h"

#include <tvm/runtime/registry.h>
//#include <tvm/te/tensor.h>
//#include <tvm/tir/expr.h>

#include "../support/scalars.h"

namespace tvm {
    PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm(DataType::Int(32), value)) {}

    PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(DataType::Float(32), value)) {}

//    PrimExpr PrimExpr::FromObject_(ObjectRef ref) {
//        using runtime::ObjectTypeChecker;
//        if (const auto* ptr = ref.as<tir::IterVarNode>()) {
//            return ptr->var;
//        }
//        if (auto opt = ref.as<te::Tensor>()) {
//            return opt.value()();
//        }
//        if (auto opt = ref.as<runtime::String>()) {
//            return tir::StringImm(opt.value());
//        }
//        if (const auto* buffer_region = ref.as<tir::BufferRegionNode>()) {
//            Array<PrimExpr> indices;
//            indices.reserve(buffer_region->region.size());
//            for (const Range& r : buffer_region->region) {
//                if (tvm::tir::is_one(r->extent)) {
//                    indices.push_back(r->min);
//                } else if (const auto* extent = r->extent.as<IntImmNode>()) {
//                    indices.push_back(tir::Ramp(r->min, tvm::tir::make_const(r->min->dtype, 1), extent->value));
//                } else {
//                    LOG(FATAL) << "ValueError: Cannot convert to BufferLoad: " << ref;
//                }
//            }
//            return tir::BufferLoad(buffer_region->buffer, indices);
//        }
//        Optional<String> actual_type = ObjectTypeChecker<PrimExpr>::CheckAndGetMismatch(ref.get());
//        ICHECK(!actual_type.defined()) << "Expected type " << ObjectTypeChecker<PrimExpr>::TypeName()
//                                       << " but got " << actual_type.value();
//        return Downcast<PrimExpr>(ref);
//    }

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