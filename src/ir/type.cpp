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

    // uint32_t t = PrimTypeNode::_GetOrAllocRuntimeTypeIndex();
    uint32_t t = TypeNode::_GetOrAllocRuntimeTypeIndex();

    // TVM_REGISTER_NODE_TYPE(PrimTypeNode);

    // TVM_REGISTER_GLOBAL("ir.PrimType")
    //         .set_body_typed([](runtime::DataType dtype) -> PrimType {
    //             return PrimType(dtype);
    //         });
}// namespace tvm
