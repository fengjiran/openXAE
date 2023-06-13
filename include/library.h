#ifndef OPENXAE_LIBRARY_H
#define OPENXAE_LIBRARY_H

//#define TVM_DLL __attribute__((visibility("default")))

template<typename T>
inline T toy_add(T a, T b) {
    return a + b;
}

template<typename T>
inline T toy_sub(T a, T b) {
    return a - b;
}

void ListGlobalFuncNames();

void test_toy_add(float a, float b);

void test_toy_sub(float a, float b);

#endif //OPENXAE_LIBRARY_H
