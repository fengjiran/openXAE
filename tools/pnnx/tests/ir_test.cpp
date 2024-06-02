//
// Created by fengj on 2024/6/1.
//
#include "glog/logging.h"
#include "pnnx/src/ir.h"
#include "gtest/gtest.h"

namespace pnnx {

TEST(IRTEST, Parameter) {
    // null parameter
    Parameter p_null;
    ASSERT_EQ(p_null.type, ParameterType::kParameterUnknown);
    ASSERT_EQ(Parameter::encode_to_string(p_null), "None");

    // bool parameter
    Parameter p_bool1(true);
    Parameter p_bool2(false);
    ASSERT_EQ(p_bool1.type, ParameterType::kParameterBool);
    ASSERT_EQ(Parameter::encode_to_string(p_bool1), "True");
    ASSERT_EQ(Parameter::encode_to_string(p_bool2), "False");

    // int parameter
    Parameter p_int1(-10ll);// 10, 10l
    std::cout << Parameter::encode_to_string(p_int1) << std::endl;
    ASSERT_EQ(p_int1.type, ParameterType::kParameterInt);
    ASSERT_EQ(Parameter::encode_to_string(p_int1), "-10");

    // float parameter
    Parameter p_float(0.3141592657);
    std::cout << Parameter::encode_to_string(p_float) << std::endl;
    ASSERT_EQ(p_float.type, ParameterType::kParameterFloat);
    ASSERT_EQ(Parameter::encode_to_string(p_float), "3.141593e-01");

    // string parameter
    Parameter p_str("pnnx");
    std::cout << Parameter::encode_to_string(p_str) << std::endl;
    ASSERT_EQ(p_str.type, ParameterType::kParameterString);
    ASSERT_EQ(Parameter::encode_to_string(p_str), "pnnx");

    // array int parameter
    Parameter p_ai1 = {1, 2, 3, 4, -5};
    ASSERT_EQ(p_ai1.type, ParameterType::kParameterArrayInt);
    ASSERT_EQ(Parameter::encode_to_string(p_ai1), "(1,2,3,4,-5)");

    Parameter p_ai2(std::vector<int>({1, 2, 3, 4, 5}));
    ASSERT_EQ(p_ai2.type, ParameterType::kParameterArrayInt);
    ASSERT_EQ(Parameter::encode_to_string(p_ai2), "(1,2,3,4,5)");

    // array float parameter
    Parameter p_af1 = {1.0, 0.112, -3.14};
    ASSERT_EQ(p_af1.type, ParameterType::kParameterArrayFloat);
    ASSERT_EQ(Parameter::encode_to_string(p_af1), "(1.000000e+00,1.120000e-01,-3.140000e+00)");

    // complex parameter
    Parameter p_c(std::complex<float>(2, 3));
    ASSERT_EQ(p_c.type, ParameterType::kParameterComplex);
    ASSERT_EQ(Parameter::encode_to_string(p_c), "2.000000e+00+3.000000e+00i");
}

}// namespace pnnx