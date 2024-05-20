//
// Created by richard on 4/29/24.
//
#include "cblas.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include <armadillo>
#include <cstdio>

TEST(ArmadilloTest, blas) {
    const int dim = 2;
    double a[4] = {1.0, 1.0, 1.0, 1.0}, b[4] = {2.0, 2.0, 2.0, 2.0}, c[4];
    int m = dim, n = dim, k = dim, lda = dim, ldb = dim, ldc = dim;
    double al = 1.0, be = 0.0;
    cblas_dgemm(static_cast<CBLAS_LAYOUT>(101),
                static_cast<CBLAS_TRANSPOSE>(111),
                static_cast<CBLAS_TRANSPOSE>(111),
                m, n, k, al, a, lda, b, ldb, be, c, ldc);
    printf("the matrix c is:%f,%f\n%f,%f\n", c[0], c[1], c[2], c[3]);
}

TEST(ArmadilloTest, test1) {
    LOG(INFO) << "ArmadilloTest test1:";
    arma::mat A(4, 5, arma::fill::randu);
    arma::mat B(4, 5, arma::fill::randu);
    LOG(INFO) << "\n"
              << A * B.t();
}

TEST(ArmadilloTest, test2) {
    LOG(INFO) << "ArmadilloTest add:";
    arma::fmat mat1 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";
    arma::fmat mat2 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    arma::fmat mat3 = "2,4,6;"
                      "8,10,12;"
                      "14,16,18";

    EXPECT_TRUE(arma::approx_equal(mat3, mat1 + mat2, "absdiff", 1e-5));
}

TEST(ArmadilloTest, test3) {
    LOG(INFO) << "ArmadilloTest sub:";
    arma::fmat mat1 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";
    arma::fmat mat2 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    arma::fmat mat3 = "0,0,0;"
                      "0,0,0;"
                      "0,0,0";
    EXPECT_TRUE(arma::approx_equal(mat3, mat1 - mat2, "absdiff", 1e-5));
}

TEST(ArmadilloTest, test4) {
    LOG(INFO) << "ArmadilloTest matmul:";
    arma::fmat mat1 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";
    arma::fmat mat2 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    arma::fmat mat3 = "30,36,42;"
                      "66,81,96;"
                      "102,126,150;";
    EXPECT_TRUE(arma::approx_equal(mat3, mat1 * mat2, "absdiff", 1e-5));
}

TEST(ArmadilloTest, test5) {
    LOG(INFO) << "ArmadilloTest pointwise:";
    arma::fmat mat1 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";
    arma::fmat mat2 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    arma::fmat mat3 = "1,4,9;"
                      "16,25,36;"
                      "49,64,81;";
    EXPECT_TRUE(arma::approx_equal(mat3, mat1 % mat2, "absdiff", 1e-5));
}

TEST(ArmadilloTest, test6) {
    auto f = [](const arma::fmat& x,
                const arma::fmat& w,
                const arma::fmat& b) -> arma::fmat {
        return x * w + b;
    };

    arma::fmat w = "1,2,3;"
                   "4,5,6;"
                   "7,8,9;";

    arma::fmat x = "1,2,3;"
                   "4,5,6;"
                   "7,8,9;";

    arma::fmat b = "1,1,1;"
                   "2,2,2;"
                   "3,3,3;";

    arma::fmat answer = "31,37,43;"
                        "68,83,98;"
                        "105,129,153";

    EXPECT_TRUE(arma::approx_equal(answer, f(x, w, b), "absdiff", 1e-5));
}

TEST(ArmadilloTest, test7) {
    auto f = [](const arma::fmat& x) -> arma::fmat {
        return arma::exp(-x);
    };

    int row = 224;
    int col = 224;

    arma::fmat x(row, col, arma::fill::randu);
    auto y = f(x);
    std::vector<float> x1(x.mem, x.mem + row * col);
    EXPECT_FALSE(y.empty());

    for (int i = 0; i < row * col; ++i) {
        EXPECT_FLOAT_EQ(std::exp(-x1[i]), y[i]);
    }
}

TEST(ArmadilloTest, test8) {
    auto f = [](const arma::fmat& x, float a, float b) -> arma::fmat {
        return a * x + b;
    };

    int row = 224;
    int col = 224;

    arma::fmat x(row, col, arma::fill::randu);

    float a = 3.0;
    float b = 4.0;
    auto y = f(x, a, b);
    std::vector<float> x1(x.mem, x.mem + row * col);
    EXPECT_FALSE(y.empty());

    for (int i = 0; i < row * col; ++i) {
        EXPECT_FLOAT_EQ(a * x1[i] + b, y[i]);
    }
}