// Copyright 2018.

#include "sapien/metrics.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace metrics {

using ::testing::DoubleEq;

TEST(AccuracyScoreMetrics, LabelTypeIsChar) {
  const size_t N = 4;
  int8_t y_true[N] = {1, 1, -1, -1};
  int8_t y_pred[N] = {-1, 1, -1, 1};

  double score = AccuracyScore(N, y_true, y_pred);

  EXPECT_THAT(score, DoubleEq(0.5));
}

TEST(AccuracyScoreMetrics, LabelTypeIsInt) {
  const size_t N = 562;
  int32_t y_true[N];
  int32_t y_pred[N];

  for (int i = 0; i < N; ++i) {
    y_true[i] = y_pred[i] = i;
  }

  double score = AccuracyScore(N, y_true, y_pred);

  EXPECT_THAT(score, DoubleEq(1.0));
}

TEST(AccuracyScoreMetrics, LabelTypeIsShort) {
  const size_t N = 573;
  int16_t y_true[N];
  int16_t y_pred[N];

  for (int i = 0; i < N; ++i) {
    y_true[i] = 1;
    y_pred[i] = -1;
  }

  EXPECT_THAT(AccuracyScore(N, y_true, y_pred), DoubleEq(0));
}
}  // namespace metrics
}  // namespace sapien
