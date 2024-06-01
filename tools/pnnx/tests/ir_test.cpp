//
// Created by fengj on 2024/6/1.
//
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "pnnx/src/ir.h"

namespace pnnx {

TEST(IRTEST, Parameter) {
    // null parameter
    Parameter p_null;
    std::cout << Parameter::encode_to_string(p_null) << std::endl;

    // bool parameter
    Parameter p_bool(true);
    std::cout << Parameter::encode_to_string(p_bool) << std::endl;

    //
}

}