// Copyright 2018

#include <vector>

#include "sapien/constants.h"
#include "sapien/sgd/classifier.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace sgd {

using ::testing::DoubleEq;

TEST(SGDClassifier, InitializeModelWithCustomOptions) {
  Classifier<int>::Options options;
  options.loss_type = HINGE_LOSS;
  options.loss_param = 1.0;
  options.learning_rate_type = LEARNING_RATE_OPTIMAL;
  options.initial_learning_rate = 0.0;
  options.penalty_type = L2_PENALTY;
  options.penalty_strength = 1e-4;
  options.l1_ratio = 0.15;
  options.max_iter = 100;
  options.fit_intercept = true;
  options.average_sgd = 0;
  options.tolerance = -Constant<double>::inf;
  options.logging_type = PER_EPOCH;
  options.fit_intercept = false;

  Classifier<int> model(options);

  const int n_samples = 4;
  const int n_features = 2;

  double X[n_samples * n_features] = {1, 1, 2, 1, -1, -1, -2, -1};
  int y[n_samples] = {1, 1, -1, -1};

  model.Train(n_samples, n_features, X, y);

  EXPECT_EQ(model.n_classes(), 2);
  EXPECT_EQ(model.n_features(), 2);

  EXPECT_THAT(model.Score(n_samples, n_features, X, y), DoubleEq(1.0));

  std::vector< std::vector<double> > coef = model.coef();
}
}  // namespace sgd
}  // namespace sapien
