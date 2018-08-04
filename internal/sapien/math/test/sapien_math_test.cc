// Copyright 2018.

#include <cmath>

#include "sapien/internal/sapien_math.h"
#include "sapien/constants.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace internal {

using ::testing::DoubleEq;
using ::testing::FloatEq;
using ::testing::ElementsAreArray;
using ::testing::ElementsAre;

TEST(SapienMath, daxpy) {
  const int N = 2;
  double alpha = 0.5;
  double X[N] = {2, 4};
  double Y[N] = {-1, 1};

  sapien_axpy(N, alpha, X, Y);

  EXPECT_THAT(Y[0], DoubleEq(0));
  EXPECT_THAT(Y[1], DoubleEq(3));
}

TEST(SapienMath, saxpy) {
  const int N = 2;
  float alpha = 0.5;
  float X[N] = {2, 4};
  float Y[N] = {-1, 1};

  sapien_axpy(N, alpha, X, Y);

  EXPECT_THAT(Y[0], FloatEq(0));
  EXPECT_THAT(Y[1], FloatEq(3));
}

TEST(SapienMath, sscal) {
  const int N = 2;
  float alpha = 0.5;
  float X[N] = {1, 2};

  sapien_scal(N, alpha, X);

  EXPECT_THAT(X[0], FloatEq(0.5));
  EXPECT_THAT(X[1], FloatEq(1));
}

TEST(SapienMath, dscal) {
  const int N = 2;
  double alpha = 0.5;
  double X[N] = {1, 2};

  sapien_scal(N, alpha, X);

  EXPECT_THAT(X[0], DoubleEq(0.5));
  EXPECT_THAT(X[1], DoubleEq(1));
}

TEST(SapienMath, sdot) {
  const int N = 2;
  float X[N] = {1, 2};
  float Y[N] = {2.5, 0.5};

  EXPECT_THAT(sapien_dot(N, X, Y), FloatEq(3.5));
}

TEST(SapienMath, ddot) {
  const int N = 2;
  double X[N] = {2, -1.5};
  double Y[N] = {1.5, 2};

  EXPECT_THAT(sapien_dot(N, X, Y), DoubleEq(0));
}

TEST(SapienMath, sgemm) {
  const int M = 2;
  const int N = 2;
  const int K = 3;

  float A[M*K] = {1, 2, 3, 4, 5, 6};    // MxK matrix, row major order.
  float B[K*N] = {0, 1, -10, 3, 9, 8};  // KxN matrix, row major order.
  float C[M*N] = {0};  // Store the result of A * B.

  float alpha = 1.0;
  float beta = 0.0;

  sapien_gemm(SAPIEN_BLAS_NO_TRANS, SAPIEN_BLAS_NO_TRANS,
              M, N, K, alpha, A, B, beta, C);

  // We expect to have C = {7, 31, 4, 67} (2x2 matrix).
  float result[4] = {7, 31, 4, 67};
  EXPECT_THAT(C, ElementsAreArray(result));
}

TEST(SapienMath, dgemm) {
  const int M = 2;
  const int N = 2;
  const int K = 3;

  double A[M*K] = {1, 2, 3, 4, 5, 6};    // MxK matrix, row major order.
  double B[K*N] = {0, 1, -10, 3, 9, 8};  // KxN matrix, row major order.
  double C[M*N] = {0};  // Store the result of A * B.

  double alpha = 1.0;
  double beta = 0.0;

  sapien_gemm(SAPIEN_BLAS_NO_TRANS, SAPIEN_BLAS_NO_TRANS,
              M, N, K, alpha, A, B, beta, C);

  // We expect to have C = {7, 31, 4, 67} (2x2 matrix).
  double result[4] = {7, 31, 4, 67};
  EXPECT_THAT(C, ElementsAreArray(result));
}

TEST(SapienMath, sgemv) {
  const int M = 6;
  const int N = 2;
  float alpha = 1.0;
  float beta = 0.0;

  float A[M*N] = {1, 2 /* 1st row */,
                  3, 4 /* 2nd row */,
                  5, 6 /* 3rd row */,
                  7, 8 /* 4th row */,
                  9, 10 /* 5th row */,
                  11, 12 /* 6th row */};
  float x[N] = {0.5 /* first row */,
                2 /* 2rd row */};

  float y[M] = {0};

  // y <- A * x.
  sapien_gemv(SAPIEN_BLAS_NO_TRANS, 6, 2, alpha, A, x, beta, y);

  float result[M] = {4.5, 9.5, 14.5, 19.5, 24.5, 29.5};

  EXPECT_THAT(y, ElementsAreArray(result));
}

TEST(SapienMath, dgemv) {
  const int M = 6;
  const int N = 2;
  double alpha = 1.0;
  double beta = 0.0;

  double A[M*N] = {1, 2 /* 1st row */,
                   3, 4 /* 2nd row */,
                   5, 6 /* 3rd row */,
                   7, 8 /* 4th row */,
                   9, 10 /* 5th row */,
                   11, 12 /* 6th row */};
  double x[N] = {0.5 /* first row */,
                 2 /* 2rd row */};

  double y[M] = {0};

  // y <- A * x.
  sapien_gemv(SAPIEN_BLAS_NO_TRANS, 6, 2, alpha, A, x, beta, y);

  double result[M] = {4.5, 9.5, 14.5, 19.5, 24.5, 29.5};

  EXPECT_THAT(y, ElementsAreArray(result));
}

TEST(SapienMath, snrm2) {
  const int N = 2;
  float X[N] = {3, 4};

  EXPECT_THAT(sapien_nrm2(N, X), FloatEq(5.0));
}

TEST(SapienMath, dnrm2) {
  const int N = 3;
  double X[N] = {3, 4, 0};

  EXPECT_THAT(sapien_nrm2(N, X), DoubleEq(5.0));
}

TEST(SapienMath, iset) {
  const int N = 734;
  int X[N];

  for (int i = 0; i < N; ++i) {
    X[i] = i + 1;
  }

  sapien_set(N, 0, X);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(X[i], 0);
  }

  sapien_set(N, 11, X);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(X[i], 11);
  }
}

TEST(SapienMath, dset) {
  const int N = 1019;
  double X[N];

  sapien_set(N, 1.25, X);

  for (int i = 0; i < N; ++i) {
    EXPECT_THAT(X[i], DoubleEq(1.25));
  }
}

TEST(SapienMath, scopy) {
  const int N = 10;
  float X[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float Y[N];

  sapien_copy(N, X, Y);

  EXPECT_THAT(Y, ElementsAreArray(X));
}

TEST(SapienMath, dcopy) {
  const int N = 10;
  double X[N] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  double Y[N];

  sapien_copy(N, X, Y);

  EXPECT_THAT(Y, ElementsAreArray(X));
}

TEST(SapienMath, xmy) {
  const int N = 4;
  float x[N] = {10, -1, 3.5, 6};
  float y[N] = {1, -3, 1.5, 0.2};
  float z[N];

  sapien_xmy(N, x, y, z);

  EXPECT_THAT(z, ElementsAre(9, 2, 2, 5.8));

  double dx[N] = {1, 2, 3, 4};
  double dy[N] = {0, -11, 34, 0};
  double dz[N];

  sapien_xmy(N, dx, dy, dz);

  EXPECT_THAT(dz, ElementsAre(1, 13, -31, 4));
}

TEST(SapienMath, xpy) {
  const int N = 4;
  float x[N] = {10, -1, 2.3, 1.9};
  float y[N] = {0.1, 0.4, 0.2, 0.1};
  float z[N];

  sapien_xpy(N, x, y, z);

  EXPECT_THAT(z, ElementsAre(10.1, -0.6, 2.5, 2));

  double dx[N] = {10, -1, 2.3, 1.9};
  double dy[N] = {0.1, 0.4, 0.2, 0.1};
  double dz[N];

  sapien_xpy(N, dx, dy, dz);

  EXPECT_THAT(dz, ElementsAre(10.1, -0.6, 2.5, 2));
}

TEST(SapienMath, vmean) {
  const int N = 100;
  float x[N];

  for (int i = 0; i < N; ++i) {
    x[i] = i;
  }

  float mean = sapien_vmean(N, x);
  EXPECT_THAT(mean, FloatEq(49.5));
}

TEST(SapienMath, isinf) {
  const double pos_inf = Constant<double>::inf;
  const double neg_inf = -Constant<double>::inf;

  EXPECT_TRUE(sapien_isinf(pos_inf));
  EXPECT_TRUE(sapien_isinf(neg_inf));

  EXPECT_FALSE(sapien_isfinite(pos_inf));
  EXPECT_FALSE(sapien_isfinite(neg_inf));
}

TEST(SapienMath, isnan) {
  const double value = Constant<double>::nan;

  EXPECT_TRUE(sapien_isnan(value));
  EXPECT_FALSE(sapien_isinf(value));
  EXPECT_FALSE(sapien_isfinite(value));

  EXPECT_TRUE(isnan(0.0 / 0.0));
  EXPECT_TRUE(isnan(Constant<double>::inf - Constant<double>::inf));
}

TEST(SapienMath, allfinite) {
  const int N = 3;
  double X[N] = {1.0, 2.0, 3.0};
  EXPECT_TRUE(sapien_allfinite(N, X));

  X[2] = Constant<double>::inf;
  EXPECT_FALSE(sapien_allfinite(N, X));

  float Y[N] = {0.0, 1.0, static_cast<float>(std::exp(800))};
  EXPECT_FALSE(sapien_allfinite(N, Y));
}
}  // namespace internal
}  // namespace sapien
