//
// Created by fengj on 2024/6/1.
//
#include "gtest/gtest.h"
#include "glog/logging.h"

int main() {
    testing::InitGoogleTest();
    google::InitGoogleLogging("pnnx_test");
    FLAGS_logtostderr = true;
    return RUN_ALL_TESTS();
}