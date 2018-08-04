// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#include "sapien/sgd/default_options.h"
#include "sapien/constants.h"

namespace sapien {
namespace sgd {

Base::Options ClassifierDefaultOptions() {
  Base::Options options;

  options.loss_type = HINGE_LOSS;
  options.loss_param = 1.0;

  options.learning_rate_type = LEARNING_RATE_OPTIMAL;
  options.initial_learning_rate = 1.0;
  options.inverse_scaling_exp = 0.5;
  options.agressiveness_param = 0.001;

  options.penalty_type = L2_PENALTY;
  options.penalty_strength = 0.0001;
  options.l1_ratio = 0.15;

  options.shuffle = true;

  options.max_iter = 10;
  options.tolerance = -Constant<double>::inf;
  options.average_sgd = 1;
  options.fit_intercept = true;

  options.logging_type = SILENT;
  options.log_to_stdout = false;

  return options;
}

Base::Options RegressorDefaultOptions() {
  Base::Options options;

  options.loss_type = SQUARED_LOSS;
  options.loss_param = 0.1;

  options.learning_rate_type = LEARNING_RATE_INVERSE_SCALING;
  options.initial_learning_rate = 0.01;
  options.inverse_scaling_exp = 0.25;
  options.agressiveness_param = 0.001;

  options.penalty_type = L2_PENALTY;
  options.penalty_strength = 0.0001;
  options.l1_ratio = 0.15;

  options.shuffle = true;

  options.max_iter = 10;
  options.tolerance = -Constant<double>::inf;
  options.average_sgd = 0;
  options.fit_intercept = true;

  options.logging_type = SILENT;
  options.log_to_stdout = false;
  return options;
}
}  // namespace sgd
}  // namespace sapien
