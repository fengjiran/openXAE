//
// Created by 赵丹 on 24-5-14.
//

#ifndef OPENXAE_DATATYPE_HPP
#define OPENXAE_DATATYPE_HPP

namespace XAcceleratorEngine {

/**
 * @brief Runtime datatype for operator attributes.
 *
 * Enumerates the datatypes supported for operator attributes like
 * weights and bias.
 */
enum class DataType {
    kTypeUnknown = 0,
    kTypeFloat32 = 1,
    kTypeFloat64 = 2,
    kTypeFloat16 = 3,
    kTypeInt32 = 4,
    kTypeInt64 = 5,
    kTypeInt16 = 6,
    kTypeInt8 = 7,
    kTypeUInt8 = 8
};

enum class ParameterType {
    kParameterUnknown = 0,
    kParameterBool = 1,
    kParameterInt = 2,

    kParameterFloat = 3,
    kParameterString = 4,
    kParameterIntArray = 5,
    kParameterFloatArray = 6,
    kParameterStringArray = 7
};

enum class StatusCode {
    kUnknownCode = -1,
    kSuccess = 0,

    kInferInputsEmpty = 1,
    kInferOutputsEmpty = 2,
    kInferParameterError = 3,
    kInferDimMismatch = 4,

    kFunctionNotImplement = 5,
    kParseWeightError = 6,
    kParseParameterError = 7,
    kParseNullOperator = 8
};

}

#endif//OPENXAE_DATATYPE_HPP
