//
// Created by richard on 6/14/23.
//

#ifndef OPENXAE_TYPE_H
#define OPENXAE_TYPE_H

#include "tvm/ir/source_map.h"
#include "tvm/node/node.h"
#include "tvm/runtime/object.h"

namespace tvm {
    /*!
    * \brief Type is the base type of all types.
    *
    * Relay's type system contains following subclasses:
    *
    * - PrimType: type of primitive type values used in the low-level IR.
    * - FuncType: type of a function.
    * - TensorType: type of certain Tensor values in the expression.
    *
    * There are also advanced types to support generic(polymorphic types).
    * \sa Type
    */
    class TypeNode : public tvm::runtime::Object {
    public:
        mutable Span span;
        static constexpr const char *_type_key = "Type";
        static constexpr const bool _type_has_method_sequal_reduce = true;
        static constexpr const bool _type_has_method_shash_reduce = true;
        static constexpr const uint32_t _type_child_slots = 14;
        TVM_DECLARE_BASE_OBJECT_INFO(TypeNode, Object);
    };

    /*!
     * \brief Managed reference to TypeNode.
     * \sa TypeNode
     */
    class Type : public tvm::runtime::ObjectRef {
    public:
        TVM_DEFINE_OBJECT_REF_METHODS(Type, tvm::runtime::ObjectRef, TypeNode);
    };

    /*!
     * \brief Primitive data types used in the low-level IR.
     *
     * PrimType represents POD-values and handles that are
     * not automatically managed by the runtime.
     *
     * \sa PrimType
     */
    class PrimTypeNode : public TypeNode {
    public:
        /*!
         * \brief The corresponding dtype field.
         */
        runtime::DataType dtype;

        void VisitAttrs(AttrVisitor *v) {
            v->Visit("dtype", &dtype);
        }

        bool SEqualReduce(const PrimTypeNode *other, SEqualReducer equal) const {
            return equal(dtype, other->dtype);
        }

        void SHashReduce(SHashReducer hash_reduce) const {
            hash_reduce(dtype);
        }

        static constexpr const char *_type_key = "PrimType";
        TVM_DECLARE_FINAL_OBJECT_INFO(PrimTypeNode, TypeNode);
    };

    /*
     * \brief Managed reference to PrimTypeNode.
     * \sa PrimTypeNode
     */
    class PrimType : public Type {
    public:
        /*!
         * \brief Constructor
         * \param dtype The corresponding dtype.
         */
        TVM_DLL explicit PrimType(runtime::DataType dtype);

        TVM_DEFINE_OBJECT_REF_METHODS(PrimType, Type, PrimTypeNode);
    };

    /*!
     * \brief Low-level raw pointer type.
     *
     *  PointerType represents type hints in the TIR to be
     *  passed to the final code generator.
     *
     *  PointerType should not occur in the high-level analysis.
     *
     * \sa PointerType
     */

    class PointerTypeNode : public TypeNode {
    public:
        /*!
         * \brief The type of the element which the pointer points to.
         */
        Type element_type;
        /*!
         * \brief The storage scope of the pointer
         */
        String storage_scope;

        void VisitAttrs(AttrVisitor *v) {
            v->Visit("element_type", &element_type);
            v->Visit("storage_scope", &storage_scope);
        }

        bool SEqualReduce(const PointerTypeNode *other, SEqualReducer equal) const {
            // Make "global" equal to ""
            String lhs_scope = storage_scope.empty() ? "global" : storage_scope;
            String rhs_scope = other->storage_scope.empty() ? "global" : other->storage_scope;
            return equal(element_type, other->element_type) && equal(lhs_scope, rhs_scope);
        }

        void SHashReduce(SHashReducer hash_reduce) const {
            hash_reduce(element_type);
            // Make "global" equal to ""
            hash_reduce(storage_scope.empty() ? "global" : storage_scope);
        }

        static constexpr const char *_type_key = "PointerType";
        TVM_DECLARE_FINAL_OBJECT_INFO(PointerTypeNode, TypeNode);
    };

    /*
     * \brief Managed reference to PointerTypeNode.
     * \sa PointerTypeNode
     */
    class PointerType : public Type {
    public:
        /*!
       * \brief Constructor
       * \param element_type The type of the element which the pointer points to.
       * \param storage_scope The storage scope into which the pointer addresses
       */
        TVM_DLL explicit PointerType(Type element_type, String storage_scope = "");

        TVM_DEFINE_OBJECT_REF_METHODS(PointerType, Type, PointerTypeNode);
    };

    /*! \brief Possible kinds of TypeVars. */
    enum TypeKind : int {
        kType = 0,
        /*! \brief Template variable in shape expression. */
        kShapeVar = 1,
        kBaseType = 2,
        kConstraint = 4,
        kAdtHandle = 5,
        kTypeData = 6
    };

    /*! \brief Converts a TypeKind to a string. */
    inline String TypeKind2String(TypeKind kind) {
        switch (kind) {
            case TypeKind::kType:
                return "Type";
            case TypeKind::kShapeVar:
                return "ShapeVar";
            case TypeKind::kBaseType:
                return "BaseType";
            case TypeKind::kConstraint:
                return "Constraint";
            case TypeKind::kAdtHandle:
                return "AdtHandle";
            case TypeKind::kTypeData:
                return "TypeData";
        }
        LOG(FATAL) << "ValueError: Unknown TypeKind: " << static_cast<int>(kind);
    }

    /*!
     * \brief Type parameter in functions.
     *
     * A type variable can be viewed as template parameter in c++ template function.
     *
     * For example, in the following pesudo code,
     * the TypeVar of f is TypeVar("n", kind=kShapeVar).
     * This function can take in a Tensor with shape=(3, 3) and
     * returns a Tensor with shape=(9,)
     *
     * \code
     *
     *  template<i32 n>
     *  f(x : Tensor[i32, (n, n)]) -> Tensor[i32, (n * n)]
     *
     * \endcode
     * \sa TypeVar, TypeKind
     */
    class TypeVarNode : public TypeNode {
    public:
        /*!
         * \brief The name of the variable,
         *  this only acts as a hint to the user,
         *  and is not used for equality.
         */
        String name_hint;
        /*! \brief The kind of type parameter */
        TypeKind kind;

        void VisitAttrs(AttrVisitor *v) {
            v->Visit("name_hint", &name_hint);
            v->Visit("kind", &kind);
            v->Visit("span", &span);
        }

        bool SEqualReduce(const TypeVarNode *other, SEqualReducer equal) const {
            return equal(kind, other->kind) && equal.FreeVarEqualImpl(this, other);
        }

        void SHashReduce(SHashReducer hash_reduce) const {
            hash_reduce(kind);
            hash_reduce.FreeVarHashImpl(this);
        }

        static constexpr const char *_type_key = "TypeVar";
        TVM_DECLARE_FINAL_OBJECT_INFO(TypeVarNode, TypeNode);
    };


}// namespace tvm

#endif// OPENXAE_TYPE_H
