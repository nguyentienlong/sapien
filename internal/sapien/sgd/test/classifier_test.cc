// Copyright 2018

#include "sapien/sgd/classifier.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace sgd {

using ::testing::DoubleEq;

TEST(Classifier, ClassifyBinaryData) {
  Classifier<int>::Options options;
  options.loss_type = LOG_LOSS;
  options.loss_param = 0.0;
  options.learning_rate_type = LEARNING_RATE_OPTIMAL;
  options.penalty_type = ELASTIC_NET_PENALTY;
  options.penalty_strength = 1e-4;
  options.l1_ratio = 0.15;
  options.max_iter = 100;
  options.fit_intercept = true;
  options.average_sgd = 1;

  Classifier<int> model(options);

  const int n_features = 2;
  const int n_samples = 8;

  double X[n_features * n_samples] = {2.0, 2.0,
                                      2.0, 1.5,
                                      1.5, 1.5,
                                      1.5, 2.0,
                                      0.0, 0.0,
                                      0.5, 0.0,
                                      0.5, 0.5,
                                      0.0, 0.5};
  double y[n_samples] = {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};

  model.Train(n_samples, n_features, X, y);
  double score = model.Score(n_samples, n_features, X, y);

  EXPECT_THAT(score, DoubleEq(1.0));
}
}  // namespace sgd
}  // namespace sapien
