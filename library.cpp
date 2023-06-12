#include <iostream>
#include "library.h"
#include "tvm/runtime/object.h"
#include "tvm/runtime/registry.h"

using TVMArgs = tvm::runtime::TVMArgs;
using TVMRetValue = tvm::runtime::TVMRetValue;
using PackedFunc = tvm::runtime::PackedFunc;

void ToyAdd(TVMArgs args, TVMRetValue* rv) {
    // automatically convert arguments to desired type.
    int a = args[0];
    int b = args[1];
}

void hello() {
    std::cout << "Hello, World!" << std::endl;
}
