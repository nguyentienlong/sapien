// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// A simple working example of SGD Regressor.

#include <iostream>

#include "sapien/sgd/regressor.h"

int main(int argc, char** argv) {
  const int n_features = 1;
  const int n_samples = 4;

  double X_train[n_samples] = {0.0, 0.5, 1.0, 2.0};
  double y_train[n_samples] = {0.0, 0.5, 1.0, 2.0};

  sapien::sgd::Regressor::Options options;
  options.loss_type = sapien::sgd::SQUARED_LOSS;
  options.learning_rate_type = sapien::sgd::LEARNING_RATE_CONSTANT;
  options.initial_learning_rate = 0.01;
  options.penalty_type = sapien::sgd::NO_PENALTY;
  options.shuffle = true;
  options.max_iter = 150;
  options.tolerance = 1e-4;
  options.logging_type = sapien::sgd::PER_EPOCH;
  options.log_to_stdout = true;

  sapien::sgd::Regressor model(options);
  model.Train(n_samples, n_features, X_train, y_train);

  const double* coef = model.coef();
  double intercept = model.intercept();

  std::cout << model.Summary() << std::endl;
  std::cout << "coef = " << *coef << std::endl;
  std::cout << "bias = " << intercept << std::endl;
  return 0;
}
