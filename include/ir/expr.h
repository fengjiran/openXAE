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

    /*!
     * \brief Base node of all primitive expressions.
     *
     *  A primitive expression deals with low-level
     *  POD data types and handles without
     *  doing life-cycle management for objects.
     *
     *  PrimExpr is used in the low-level code
     *  optimizations and integer analysis.
     *
     * \sa PrimExpr
     */
    class PrimExprNode : public BaseExprNode {
    public:
        /*!
         * \brief The runtime data type of the primitive expression.
         *
         * runtime::DataType(dtype) provides coarse grained type information
         * during compile time and runtime. It is eagerly built in
         * PrimExpr expression construction and can be used for
         * quick type checking.
         *
         * dtype is sufficient to decide the Type of the PrimExpr
         * when it corresponds to POD value types such as i32.
         *
         * When dtype is DataType::Handle(), the expression could corresponds to
         * a more fine-grained Type, and we can get the type by running lazy type inference.
         */
        DataType dtype;

        TVM_OBJECT_ENABLE_SCRIPT_PRINTER();

        static constexpr const char* _type_key = "PrimExpr";
        static constexpr const uint32_t _type_child_slots = 38;
        TVM_DECLARE_BASE_OBJECT_INFO(PrimExprNode, BaseExprNode);
    };

    /*!
     * \brief Reference to PrimExprNode.
     * \sa PrimExprNode
     */
    class PrimExpr : public BaseExpr {
    public:
        /*!
         * \brief construct from integer.
         * \param value The value to be constructed.
         */
        TVM_DLL PrimExpr(int32_t value);// NOLINT(*)
        /*!
         * \brief construct from float.
         * \param value The value to be constructed.
         */
        TVM_DLL PrimExpr(float value);// NOLINT(*)

        /*! \return the data type of this expression. */
        DataType dtype() const { return static_cast<const PrimExprNode*>(get())->dtype; }

        TVM_DEFINE_OBJECT_REF_METHODS(PrimExpr, BaseExpr, PrimExprNode);

    private:
        // Internal function for conversion.
        friend struct runtime::PackedFuncValueConverter<PrimExpr>;
        TVM_DLL static PrimExpr FromObject_(ObjectRef ref);
    };

    /*!
     * \brief Constant integer literals in the program.
     * \sa IntImm
     */
    class IntImmNode : public PrimExprNode {
    public:
        /*! \brief the Internal value. */
        int64_t value;

        void VisitAttrs(AttrVisitor* v) {
            v->Visit("dtype", &dtype);
            v->Visit("value", &value);
            v->Visit("span", &span);
        }

        bool SEqualReduce(const IntImmNode* other, SEqualReducer equal) const {
            return equal(dtype, other->dtype) && equal(value, other->value);
        }

        void SHashReduce(SHashReducer hash_reduce) const {
            hash_reduce(dtype);
            hash_reduce(value);
        }

        static constexpr const char* _type_key = "IntImm";
        TVM_DECLARE_FINAL_OBJECT_INFO(IntImmNode, PrimExprNode);
    };

    /*!
     * \brief Managed reference class to IntImmNode.
     *
     * \sa IntImmNode
     */
    class IntImm : public PrimExpr {
    public:
        /*!
         * \brief Constructor.
         * \param dtype The data type of the value.
         * \param value The internal value.
         * \param span The location of this object in the source code.
         */
        TVM_DLL IntImm(DataType dtype, int64_t value, Span span = Span());

        TVM_DEFINE_OBJECT_REF_METHODS(IntImm, PrimExpr, IntImmNode);
        TVM_DEFINE_OBJECT_REF_COW_METHOD(IntImmNode);
    };

}// namespace tvm

#endif//OPENXAE_EXPR_H
