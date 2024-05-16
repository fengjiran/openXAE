//
// Created by 赵丹 on 24-5-16.
//

#ifndef OPENXAE_PARAMETER_HPP
#define OPENXAE_PARAMETER_HPP

namespace XAcceleratorEngine {

/**
 * 计算节点中的参数信息，参数一共可以分为如下的几类
 * 1.int
 * 2.float
 * 3.string
 * 4.bool
 * 5.int array
 * 6.string array
 * 7.float array
 */

class Parameter {
public:
    explicit Parameter(ParameterType type_ = ParameterType::kParameterUnknown)
        : type(type_) {}

    virtual ~Parameter() = default;

    ParameterType type = ParameterType::kParameterUnknown;
};

class ParameterInt : public Parameter {
public:
    ParameterInt() : Parameter(ParameterType::kParameterInt) {}

    explicit ParameterInt(int32_t value_)
        : Parameter(ParameterType::kParameterInt), value(value_) {}

    int32_t value = 0;
};

class ParameterFloat : public Parameter {
public:
    ParameterFloat() : Parameter(ParameterType::kParameterFloat) {}
    explicit ParameterFloat(float value_)
        : Parameter(ParameterType::kParameterFloat), value(value_) {}

    float value = 0.f;
};

}// namespace XAcceleratorEngine

#endif//OPENXAE_PARAMETER_HPP
