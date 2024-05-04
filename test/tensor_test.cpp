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
    Tensor<float> f1(10, 5, 5);
    f1.RandN();
    Tensor<float> f2(f1);
    EXPECT_EQ(f2.GetChannels(), 10);
    EXPECT_EQ(f2.GetRows(), 5);
    EXPECT_EQ(f2.GetCols(), 5);
    EXPECT_TRUE(arma::approx_equal(f1.data(), f2.data(), "absdiff", 1e-4));

    Tensor<float> f3(3, 2, 1);
    f3 = f1;
    EXPECT_EQ(f3.GetChannels(), 10);
    EXPECT_EQ(f3.GetRows(), 5);
    EXPECT_EQ(f3.GetCols(), 5);
    EXPECT_TRUE(arma::approx_equal(f1.data(), f3.data(), "absdiff", 1e-4));
}

TEST(TensorTest, moveCtor) {
    Tensor<float> t1(5, 5);
    Tensor<float> t2(3, 4);
    t2 = std::move(t1);

    EXPECT_EQ(t2.GetChannels(), 1);
    EXPECT_EQ(t2.GetRows(), 5);
    EXPECT_EQ(t2.GetRows(), 5);

    EXPECT_EQ(t1.data().memptr(), nullptr);
}

TEST(TensorTest, setData) {
    Tensor<float> t1(5, 10, 10);
    arma::fcube cube(10, 10, 5);
    cube.randn();
    t1.SetData(cube);
    EXPECT_TRUE(arma::approx_equal(t1.data(), cube, "absdiff", 1e-4));
}

TEST(TensorTest, transform1) {
    Tensor<float> t1(5, 10, 10);
    t1.Fill(1.0f);
    t1.Transform([](const float& value) { return value * 2.0f; });
    for (size_t i = 0; i < t1.GetSize(); ++i) {
        EXPECT_EQ(t1.index(i), 2.0f);
    }
}

TEST(TensorTest, clone) {
    std::shared_ptr<Tensor<float>> t1 = std::make_shared<Tensor<float>>(3, 3, 3);
    EXPECT_FALSE(t1->empty());
    t1->RandN();

    const auto& t2 = CloneTensor(t1);
    EXPECT_TRUE(t1->RawPtr() != t2->RawPtr());
    EXPECT_EQ(t1->GetSize(), t2->GetSize());
    for (size_t i = 0; i < t1->GetSize(); ++i) {
        EXPECT_FLOAT_EQ(t1->index(i), t2->index(i));
    }
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