// Copyright 2018.

#include "sapien/utility/weight_vector.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace internal {

using ::testing::DoubleEq;
using ::testing::FloatEq;
using ::testing::ElementsAre;

TEST(WeightVector, ScaleFloatVectorByAScalar) {
  const int N = 3;
  float v[N] = {1, 2, 3};

  WeightVector<float> weight(N, v);
  EXPECT_EQ(weight.n_elem, N);
  weight.Scale(2.0);

  // Before calling Reset:
  EXPECT_THAT(v, ElementsAre(1, 2, 3));

  weight.Reset();

  // Now we expect v to be scaled up by 2.
  EXPECT_THAT(v, ElementsAre(2, 4, 6));
}

TEST(WeightVector, ScaleDoubleVectorByAScalar) {
  const int N = 3;
  double v[N] = {1, 2, 3};

  WeightVector<double> weight(N, v);
  EXPECT_EQ(weight.n_elem, N);
  weight.Scale(2.0);

  // Before calling Reset:
  EXPECT_THAT(v, ElementsAre(1, 2, 3));

  weight.Reset();

  // Now we expect v to be scaled up by 2.
  EXPECT_THAT(v, ElementsAre(2, 4, 6));
}

TEST(WeightVector, AddWithAnotherVector) {
  const int N = 3;
  double a[N] = {1, 2, 3};
  double x[N] = {-1, 0, 2};

  WeightVector<double> weight(N, a);

  // w = w + 2.0 * x
  weight.Add(x, 2.0);

  // Before calling reset.
  EXPECT_THAT(a, ElementsAre(-1, 2, 7));

  weight.Reset();

  // After calling reset.
  EXPECT_THAT(a, ElementsAre(-1, 2, 7));
}

TEST(WeightVector, ScaleVectorByScaleAndAddWithAnotherVector) {
  const int N = 3;
  double a[N] = {1, 2, 3};
  double x[N] = {-1, 0, 2};

  WeightVector<double> weight(N, a);
  weight.Scale(2.0);
  weight.Add(x, 4.0);

  // Before calling Reset, we would expect a = a + 2x.
  EXPECT_THAT(a, ElementsAre(-1, 2, 7));

  weight.Reset();

  // Now we expect that a = 2*a + 4*x = 2(a + 2x)
  EXPECT_THAT(a, ElementsAre(-2, 4, 14));
}

TEST(WeightVector, DotWithAnotherVector) {
  const int N = 3;
  double a[N] = {1, 2, 3};
  double x[N] = {-1, 0, 2};

  WeightVector<double> weight(N, a);
  weight.Scale(2.0);

  EXPECT_THAT(weight.Dot(x), DoubleEq(10.0));

  weight.Reset();

  EXPECT_THAT(weight.Dot(x), DoubleEq(10.0));
  EXPECT_THAT(a, ElementsAre(2, 4, 6));
}

TEST(WeightVector, AddAverage) {
  const int N = 2;
  double x[N] = {0, -1};
  double a[N] = {1, 2};
  double aw[N] = {0, 0};

  WeightVector<double> weight(N, a, aw);

  for (int i = 1; i <= 4; ++i) {
    // weight += 2 * x
    weight.Add(x, 2);
    weight.AddAverage(x, 2, i);
  }

  weight.Reset();

  double avg1 = 1.0;
  double avg2 = (0.0 - 2.0 - 4.0 - 6.0) / 4.0;

  EXPECT_THAT(aw[0], DoubleEq(avg1));
  EXPECT_THAT(aw[1], DoubleEq(avg2));

  weight.Reset();

  EXPECT_THAT(aw[0], DoubleEq(avg1));
  EXPECT_THAT(aw[1], DoubleEq(avg2));
}

TEST(WeightVector, AddAverage2) {
  const int N = 2;
  float w0[N] = {1, 2};
  float aw[N] = {0, 0};

  WeightVector<float> weight(N, w0, aw);

  // w1 = 2 * w0 + (4, 0) = (6, 4)
  float x1[N] = {4, 0};
  weight.Scale(2.0);
  weight.Add(x1, 1.0);
  weight.AddAverage(x1, 1.0, 1);

  // w2 = 0.5 * w1 = (3, 2)
  float x2[N] = {0, 0};
  weight.Scale(0.5);
  weight.AddAverage(x2, 1.0, 2);

  // w3 = w2 + (0, 1).
  float x3[N] = {0, 1};
  weight.Add(x3, 1.0);
  weight.AddAverage(x3, 1.0, 3);

  weight.Reset();

  // aw = (w1 + w2 + w3) / 3.
  EXPECT_THAT(aw[0], FloatEq(4.0));
  EXPECT_THAT(aw[1], FloatEq(3.0));
}

TEST(WeightVector, ComputeL1Norm) {
  const int N = 2;
  double w0[N] = {3, 4};

  WeightVector<double> weight(N, w0);
  weight.Scale(2.0);

  EXPECT_THAT(weight.L2Norm(), DoubleEq(10.0));

  weight.Reset();
  EXPECT_THAT(weight.L2Norm(), DoubleEq(10.0));
}
}  // namespace internal
}  // namespace sapien
