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

}

#endif//OPENXAE_DATATYPE_HPP
