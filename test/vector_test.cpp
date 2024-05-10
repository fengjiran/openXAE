//
// Created by richard on 5/4/24.
//

#include "MyVector.hpp"
#include "gtest/gtest.h"

namespace XAcceleratorEngine {

TEST(MyVectorTest, general) {
    vec<int> v{8, 4, 5, 9};
    v.push_back(6);
    v.push_back(9);
    v[2] = -1;

    vec<int> ans{8, 4, -1, 9, 6, 9};

    ASSERT_TRUE(v == ans);
}

TEST(MyVectorTest, back) {
    vec<char> letters{'a', 'b', 'c', 'd', 'e', 'f'};
    std::cout << letters;
    ASSERT_EQ(letters.back(), 'f');
    if (!letters.empty())
        std::cout << "The last character is '" << letters.back() << "'.\n";
}

}// namespace XAcceleratorEngine
