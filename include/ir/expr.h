//
// Created by richard on 6/20/23.
//

#ifndef OPENXAE_EXPR_H
#define OPENXAE_EXPR_H

#include <ir/source_map.h>
#include <ir/type.h>
#include <node/node.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>

namespace tvm {
    /*!
     * \brief Base type of all the expressions.
     * \sa Expr
     */
    class BaseExprNode : public Object {
    public:
        /*!
         * \brief Span that points to the original source code.
         *        Reserved debug information.
         */
        mutable Span span;

        static constexpr const char* _type_key = "BaseExpr";
        static constexpr const bool _type_has_method_sequal_reduce = true;
        static constexpr const bool _type_has_method_shash_reduce = true;
        static constexpr const uint32_t _type_child_slots = 62;
        TVM_DECLARE_BASE_OBJECT_INFO(BaseExprNode, Object);
    };

    /*!
     * \brief Managed reference to BaseExprNode.
     * \sa BaseExprNode
     */
    class BaseExpr : public ObjectRef {
    public:
        TVM_DEFINE_OBJECT_REF_METHODS(BaseExpr, ObjectRef, BaseExprNode);
    };
}// namespace tvm

#endif//OPENXAE_EXPR_H
