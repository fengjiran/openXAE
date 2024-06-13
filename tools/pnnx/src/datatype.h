//
// Created by 赵丹 on 24-6-14.
//

#ifndef OPENXAE_DATATYPE_H
#define OPENXAE_DATATYPE_H

namespace pnnx {

/**
 * @brief Runtime parameter type.
 *
 * Enumerates the parameter type_ supported for workload.
 */
enum class ParameterType {
    kParameterUnknown = 0,
    kParameterBool = 1,
    kParameterInt = 2,
    kParameterFloat = 3,
    kParameterString = 4,
    kParameterArrayInt = 5,
    kParameterArrayFloat = 6,
    kParameterArrayString = 7,
    kParameterComplex = 10,
    kParameterArrayComplex = 11
};

/**
 * @brief Runtime data type.
 *
 * Enumerates the data type supported for workload.
 * <p>
 * 0 = null \n
 * 1 = float32 \n
 * 2 = float64 \n
 * 3 = float16 \n
 * 4 = int32 \n
 * 5 = int64 \n
 * 6 = int16 \n
 * 7 = int8 \n
 * 8 = uint8 \n
 * 9 = bool \n
 * 10 = complex64 \n
 * 11 = complex128 \n
 * 12 = complex32 \n
 * 13 = bf16
 */
enum class DataType {
    kDataTypeUnknown = 0,
    kDataTypeFloat32 = 1,
    kDataTypeFloat64 = 2,
    kDataTypeFloat16 = 3,
    kDataTypeInt32 = 4,
    kDataTypeInt64 = 5,
    kDataTypeInt16 = 6,
    kDataTypeInt8 = 7,
    kDataTypeUInt8 = 8,
    kDataTypeBool = 9,
    kDataTypeComplex64 = 10,
    kDataTypeComplex128 = 11,
    kDataTypeComplex32 = 12,
    kDataTypeBFloat16 = 13
};

}// namespace pnnx

#endif//OPENXAE_DATATYPE_H
