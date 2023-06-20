//
// Created by richard on 6/14/23.
//

/*!
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include "ir/type.h"

#include "tvm/runtime/registry.h"

namespace tvm {
    PrimType::PrimType(runtime::DataType dtype) {
        ObjectPtr<PrimTypeNode> n = make_object<PrimTypeNode>();
        n->dtype = dtype;
        data_ = std::move(n);
    }
    TVM_REGISTER_NODE_TYPE(PrimTypeNode);
    TVM_REGISTER_GLOBAL("ir.PrimType")
            .set_body_typed([](runtime::DataType dtype) -> PrimType {
                return PrimType(dtype);
            });

    PointerType::PointerType(Type element_type, String storage_scope) {
        ObjectPtr<PointerTypeNode> n = make_object<PointerTypeNode>();
        n->element_type = std::move(element_type);
        n->storage_scope = std::move(storage_scope);
        data_ = std::move(n);
    }
    TVM_REGISTER_NODE_TYPE(PointerTypeNode);
    TVM_REGISTER_GLOBAL("ir.PointerType")
            .set_body_typed([](Type element_type, String storage_scope = "") -> PointerType {
                return PointerType(element_type, storage_scope);
            });

}// namespace tvm
