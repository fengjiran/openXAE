#include <iostream>
#include "include/library.h"
#include "tvm/runtime/registry.h"

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("toy_add").set_body([](TVMArgs args, TVMRetValue *rv) -> void {
    *rv = toy_add<double>(args[0], args[1]);
});

void ListGlobalFuncNames() {
    int global_func_num;
    const char** plist;
    TVMFuncListGlobalNames(&global_func_num, &plist);

    LOG_INFO << "List all " << global_func_num << " global functions:";
    for (int i = 0; i < global_func_num; i++) {
        std::cout << plist[i] << std::endl;
    }
}
