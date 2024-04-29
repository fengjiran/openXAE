//
// Created by richard on 6/20/23.
//

#ifndef OPENXAE_TENSOR_TYPE_H
#define OPENXAE_TENSOR_TYPE_H

#include "ir/expr.h"
#include "ir/type.h"

namespace tvm {
    /*!
     * \brief Base of all Tensor types
     *  This container can hold TensorType or GenericTensorType.
     * \sa BaseTensorType, TensorTypeNode
     */
    class BaseTensorTypeNode : public TypeNode {
    public:
        static constexpr const char* _type_key = "relay.BaseTensorType";
        static constexpr const uint32_t _type_child_slots = 1;
        TVM_DECLARE_BASE_OBJECT_INFO(BaseTensorTypeNode, TypeNode);
    };

    /*!
     * \brief Managed reference to BaseTensorTypeNode.
     * \sa BaseTensorTypeNode.
     */
    class BaseTensorType : public Type {
    public:
        TVM_DEFINE_OBJECT_REF_METHODS(BaseTensorType, Type, BaseTensorTypeNode);
    };

    /*!
     * \brief This is the most commonly used type in relay.
     *  TensorType have a fixed dimension, data type.
     *
     *  The elements of shape can be either IntImm(constant integer),
     *  or any symbolic integer expression.
     *  The symbolic integer allows generic shape inference in certain cases.
     * \sa TensorType
     */
    class TensorTypeNode : public BaseTensorTypeNode {
    public:
        /*!
       * \brief The shape of the tensor,
       *  represented by PrimExpr(tvm::Expr).
       */
        Array<PrimExpr> shape;
        /*! \brief The content data type */
        DataType dtype;

        void VisitAttrs(tvm::AttrVisitor* v) {
            v->Visit("shape", &shape);
            v->Visit("dtype", &dtype);
            v->Visit("span", &span);
        }

        bool SEqualReduce(const TensorTypeNode* other, SEqualReducer equal) const {
            return equal(shape, other->shape) && equal(dtype, other->dtype);
        }

        void SHashReduce(SHashReducer hash_reduce) const {
            hash_reduce(shape);
            hash_reduce(dtype);
        }

        /*! \brief Return product of elements in the shape.
         *  \return (d1 * d_2 ... * d_n) if shape is (d_1, d_2, ..., d_n) and 1 if shape size is zero.
         */
        TVM_DLL PrimExpr Size() const;

        static constexpr const char* _type_key = "relay.TensorType";
        TVM_DECLARE_FINAL_OBJECT_INFO(TensorTypeNode, BaseTensorTypeNode);
    };

    /*!
     * \brief Managed reference to TensorTypeNode.
     * \sa TensorTypeNode.
     */
    class TensorType : public Type {
    public:
        /*!
         * \brief Constructor.
         * \param shape The shape of the tensor.
         * \param dtype The runtime dtype of the tensor's elements.
         */
        TVM_DLL TensorType(Array<PrimExpr> shape, DataType dtype);

        /*!
         * \brief Construct an scalar containing elements of dtype.
         * \param dtype The runtime dtype of the tensor's elements.
         * \return THe constructed type.
         */
        TVM_DLL static TensorType Scalar(DataType dtype);

        TVM_DEFINE_OBJECT_REF_METHODS(TensorType, Type, TensorTypeNode);
    };
}// namespace tvm

#endif//OPENXAE_TENSOR_TYPE_H
