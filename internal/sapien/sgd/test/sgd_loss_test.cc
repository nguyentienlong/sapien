// Copyright 2018.

#include "sapien/sgd/loss.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace sgd {

using ::testing::DoubleEq;
using ::testing::FloatEq;

TEST(SGDLoss, ModifiedHuberLoss) {
  ModifiedHuberLoss<double> loss;

  // z = p * y >= 1.0
  double y1 = 1.0;
  double p1 = 1.25;

  EXPECT_THAT(loss(p1, y1), DoubleEq(0.0));
  EXPECT_THAT(loss.FirstDerivative(p1, y1), DoubleEq(0.0));

  // -1.0 < z = p * y < 1.0
  double y2 = -1.0;
  double p2 = 0.25;
  double z2 = p2 * y2;
  double true_loss_2 = (1.0 - z2) * (1.0 - z2);
  double true_dloss_2 = 2.0 * (1.0 - z2) * (-y2);

  EXPECT_THAT(loss(p2, y2), true_loss_2);
  EXPECT_THAT(loss.FirstDerivative(p2, y2), true_dloss_2);

  // z = p * y<= -1.0
  double y3 = -1;
  double p3 = 5.4372382834981;
  double z3 = p3 * y3;
  double true_loss_3 = -4.0 * z3;
  double true_dloss_3 = -4.0 * y3;

  EXPECT_THAT(loss(p3, y3), DoubleEq(true_loss_3));
  EXPECT_THAT(loss.FirstDerivative(p3, y3), DoubleEq(true_dloss_3));
}

TEST(SGDLoss, HingeLoss) {
  HingeLoss<double> loss(1.0);

  // z = p * y <= 1.0 = threshold.
  double y1 = -1;
  double p1 = -0.23273782;
  double z1 = p1 * y1;
  double true_l1 = 1.0 - z1;
  double true_dl1 = -y1;

  EXPECT_THAT(loss(p1, y1), DoubleEq(true_l1));
  EXPECT_THAT(loss.FirstDerivative(p1, y1), DoubleEq(true_dl1));

  // z = p * y > 1.0 = threshold.
  double y2 = 1;
  double p2 = 1.02323;

  EXPECT_THAT(loss(p2, y2), DoubleEq(0.0));
  EXPECT_THAT(loss.FirstDerivative(p2, y2), DoubleEq(0.0));
}

TEST(SGDLoss, SquaredHingeLoss) {
  SquaredHingeLoss<double> loss(1.0);

  // z = p * y > 0.0
  double y1 = 1;
  double p1 = 0.47261362;
  double z1 = 1.0 - p1 * y1;

  EXPECT_THAT(loss(p1, y1), DoubleEq(z1 * z1));
  EXPECT_THAT(loss.FirstDerivative(p1, y1), DoubleEq(-2.0 * y1 * z1));
}
}  // namespace sgd
}  // namespace sapien
