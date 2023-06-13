#ifndef OPENXAE_LIBRARY_H
#define OPENXAE_LIBRARY_H

//#define TVM_DLL __attribute__((visibility("default")))

template<typename T>
inline T toy_add(T a, T b) {
    return a + b;
}

void ListGlobalFuncNames();

#endif //OPENXAE_LIBRARY_H
