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

  void VisitAttrs(AttrVisitor *v) { v->Visit("dtype", &dtype); }

  bool SEqualReduce(const PrimTypeNode *other, SEqualReducer equal) const {
    return equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dtype); }

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

} // namespace tvm

#endif // OPENXAE_TYPE_H
