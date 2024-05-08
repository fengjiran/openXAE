//
// Created by richard on 5/4/24.
//

#include "MyVector.hpp"
#include "gtest/gtest.h"

namespace XAcceleratorEngine {

TEST(MyVectorTest, general) {
    vec<int> v;
    v = {8, 4, 5, 9};
    v.push_back(6);
    v.push_back(9);
    v[2] = -1;

    for (int n: v) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

TEST(MyVectorTest, back) {
    vec<char> letters{'a', 'b', 'c', 'd', 'e', 'f'};
    if (!letters.empty())
        std::cout << "The last character is '" << letters.back() << "'.\n";
}

}// namespace XAcceleratorEngine
