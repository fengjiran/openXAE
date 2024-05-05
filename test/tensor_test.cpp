//
// Created by richard on 4/28/24.
//

#include "Tensor.hpp"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace XAcceleratorEngine {

TEST(TensorTest, init1D) {
    Tensor<float> f1(3);// row vector
    const auto& rawShape = f1.GetRawShape();
    EXPECT_EQ(rawShape.size(), 1);
    EXPECT_EQ(rawShape[0], 3);
}

TEST(TensorTest, init2D) {
    Tensor<float> f1(10, 25);
    const auto& rawShape1 = f1.GetRawShape();
    EXPECT_EQ(rawShape1.size(), 2);
    EXPECT_EQ(rawShape1[0], 10);
    EXPECT_EQ(rawShape1[1], 25);

    Tensor<float> f2(1, 25);
    const auto& rawShape2 = f2.GetRawShape();
    EXPECT_EQ(rawShape2.size(), 1);
    EXPECT_EQ(rawShape2[0], 25);
}

TEST(TensorTest, init3D) {
    Tensor<float> f1(3, 224, 224);
    EXPECT_EQ(f1.GetChannels(), 3);
    EXPECT_EQ(f1.GetRows(), 224);
    EXPECT_EQ(f1.GetCols(), 224);
    EXPECT_EQ(f1.GetSize(), 3 * 224 * 224);
    EXPECT_EQ(f1.GetRawShape().size(), 3);
    EXPECT_EQ(f1.GetRawShape()[0], 3);

    Tensor<float> f2(std::vector<uint32_t>{1, 13, 14});
    EXPECT_EQ(f2.GetChannels(), 1);
    EXPECT_EQ(f2.GetRows(), 13);
    EXPECT_EQ(f2.GetCols(), 14);
    EXPECT_EQ(f2.GetSize(), 13 * 14);
}

TEST(TensorTest, copyCtor) {
    Tensor<float> f1(5, 5);
    f1.Fill(2.0);

    Tensor<float> f2(f1);
    Tensor<float> f3 = f1;
    f1.Show();
    f2.Show();
    f3.Show();
}

TEST(TensorTest, moveCtor) {
    Tensor<float> f1(5, 5);
    f1.Fill(3.0);

    Tensor<float> f2(std::move(f1));
    CHECK(f1.empty());
    f2.Show();

    Tensor<float> f3 = std::move(f2);
    CHECK(f2.empty());
    f3.Show();
}

TEST(TensorTest, review1) {
    std::cout << "TensorTest_review1\n";
    size_t totalElems = 10;
    size_t rows = 5;
    size_t cols = totalElems / rows;
    CHECK_EQ(totalElems, rows * cols);
    std::vector<float> vec(totalElems);
    std::iota(vec.begin(), vec.end(), 0);
    std::cout << "original data: \n";
    for (auto& it: vec) {
        std::cout << it << " ";
    }

    Tensor<float> f1(rows, cols);
    f1.Fill(vec, true);
    f1.Show();
    std::cout << "\ndata after review: \n";
    for (size_t i = 0; i < totalElems; ++i) {
        std::cout << *(f1.RawPtr() + i) << " ";
    }
}

}// namespace XAcceleratorEngine