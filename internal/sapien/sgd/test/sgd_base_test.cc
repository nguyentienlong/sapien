// Copyright 2018.

#include "sapien/sgd/base.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace sgd {

TEST(SGDBase, InitializeBaseModel) {
  Base::Options options;
  options.loss_type = HINGE_LOSS;
  options.loss_param = 1.0;
  options.learning_rate_type = LEARNING_RATE_OPTIMAL;
  options.penalty_type = L2_PENALTY;
  options.penalty_strength = 1e-4;
  options.l1_ratio = 0.15;
  options.max_iter = 100;
  options.fit_intercept = false;
  options.average_sgd = 0;

  Base model(CLASSIFICATION_MODEL, options);

  const int n_features = 2;
  const int n_samples = 4;

  double X[n_features * n_samples] = {1.0, 1.0,
                                      2.0, 1.0,
                                      -1.0, -1.0,
                                      -2.0, -1.0};
  double y[n_samples] = {1.0, 1.0, -1.0, -1.0};
  double sample_weight[n_samples] = {1.0, 1.0, 1.0, 1.0};

  double weight[n_features] = {0};
  double intercept = 0.0;
  double average_weight[n_features] = {0};
  double average_intercept = 0.0;

  model.TrainOne(n_samples, n_features, X, y, sample_weight,
                 weight, &intercept,
                 average_weight, &average_intercept,
                 1.0, 1.0);
}
}  // namespace sgd
}  // namespace sapien
