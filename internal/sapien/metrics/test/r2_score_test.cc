// Copyright 2018.

#include "sapien/metrics.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace metrics {

using ::testing::DoubleEq;
using ::testing::FloatEq;

TEST(R2ScoreMetrics, PredictedVectorIsInverseOfTrueVector) {
  const int N = 3;
  float y_true[N] = {1, 2, 3};
  float y_pred[N] = {3, 2, 1};
  float score = R2Score(N, y_true, y_pred);
  EXPECT_THAT(score, FloatEq(-3.0));

  double dy_true[N] = {1, 2, 3};
  double dy_pred[N] = {3, 2, 1};
  double dscore = R2Score(N, dy_true, dy_pred);
  EXPECT_THAT(dscore, DoubleEq(-3.0));
}

TEST(R2ScoreMetrics, PredictedVectorIsExactTheSameAsTrueVector) {
  const int N = 4;
  float y_true[N] = {0.1, 0.2, 0.3, 0.4};
  float y_pred[N] = {0.1, 0.2, 0.3, 0.4};
  float score = R2Score(N, y_true, y_pred);
  EXPECT_THAT(score, FloatEq(1.0));

  double dy_true[N] = {0.1, 0.2, 0.3, 0.4};
  double dy_pred[N] = {0.1, 0.2, 0.3, 0.4};
  double dscore = R2Score(N, dy_true, dy_pred);
  EXPECT_THAT(dscore, DoubleEq(1.0));
}

TEST(R2ScoreMetrics, PredictedVectorIsArbitrary) {
  const int N = 4;
  float y_true[N] = {3, -0.5, 2, 7};
  float y_pred[N] = {2.5, 0.0, 2, 8};
  float score = R2Score(N, y_true, y_pred);
  EXPECT_NEAR(score, 0.948, 1e-3);

  double dy_true[N] = {3, -0.5, 2, 7};
  double dy_pred[N] = {2.5, 0.0, 2, 8};
  double dscore = R2Score(N, dy_true, dy_pred);
  EXPECT_NEAR(dscore, 0.948, 1e-3);
}
}  // namespace metrics
}  // namespace sapien
