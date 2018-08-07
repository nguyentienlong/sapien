// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <cstring>  // memcpy

#include "sapien/sgd/sgd_regressor.h"
#include "sapien/sgd/default_options.h"
#include "sapien/metrics.h"
#include "sapien/internal/sapien_math.h"
#include "sapien/utility/wall_time.h"
#include "sapien/utility/stringprintf.h"
#include "glog/logging.h"

namespace sapien {
namespace sgd {

using internal::sapien_gemv;
using internal::sapien_set;

using internal::WallTimeInSeconds;
using internal::StringPrintf;

using metrics::R2Score;

// Default constructor
SGDRegressor::SGDRegressor()
    : Base(RegressorDefaultOptions()),
      summary_("SGDRegressor: model has not been trained yet"),
      n_features_(0),
      coef_(nullptr),
      intercept_(0.0) {
  // We use squared loss as the default loss functor for regression model.
  Base::loss_functor(new SquaredLoss<double>());
}

// Constructor that takes custom options provided by user.
SGDRegressor::SGDRegressor(const SGDRegressor::Options& options)
    : Base(options),
      summary_("SGDRegressor: model has not been trained yet"),
      n_features_(0),
      coef_(nullptr),
      intercept_(0.0) {
  // We use squared loss as the default loss functor for regression model.
  Base::loss_functor(new SquaredLoss<double>());
}

// Train
void SGDRegressor::Train(const size_t n_samples,
                         const size_t n_features,
                         const double* X,
                         const double* y) {
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  CHECK_NOTNULL(this->loss_functor());
  CHECK(!this->loss_functor()->IsClassification())
      << "Invalid loss functor for regression model";

  double start = WallTimeInSeconds();

  n_features_ = n_features;

  // Initialize coef_ vector
  // TODO(Linh): This should really be changed in order to allow client
  //             to initialize coef matrix another way.
  coef_.reset(new double[n_features]);
  sapien_set(n_features, 0.0, coef_.get());

  // Initialize sample weights.
  // For the time being, all sample weights are initialized to 0.0s
  // TODO(Linh): Brainstorm of other methods to initialize sample weights.
  //             (also see Classifier).
  std::unique_ptr<double[]> sample_weight(new double[n_samples]);
  sapien_set(n_samples, 1.0, sample_weight.get());

  std::unique_ptr<double[]> average_weight;
  double average_intercept = 0.0;

  const bool fit_average_sgd = this->options().average_sgd > 0;

  if (fit_average_sgd) {
    average_weight = std::unique_ptr<double[]>(new double[n_features]);
  }

  this->TrainOne(n_samples,
                 n_features,
                 X,
                 y,
                 sample_weight.get(),
                 coef_.get(),
                 &intercept_,
                 average_weight.get(),
                 &average_intercept,
                 1.0,
                 1.0);

  if (fit_average_sgd) {
    std::memcpy(coef_.get(), average_weight.get(), n_features * sizeof(double));
    intercept_ = average_intercept;
  }

  // Summary
  summary_ = StringPrintf("Training time: %.6f, Training score: %.6f",
                          WallTimeInSeconds() - start,
                          Score(n_samples, n_features, X, y));
}

// Score
double SGDRegressor::Score(const size_t n_samples,
                           const size_t n_features,
                           const double* X, const double* y) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);
  CHECK_EQ(n_features, n_features_);

  std::unique_ptr<double[]> y_pred(new double[n_samples]);
  Predict(n_samples, n_features, X, y_pred.get());
  return R2Score(n_samples, y, y_pred.get());
}

// Predict
void SGDRegressor::Predict(const size_t n_samples,
                           const size_t n_features,
                           const double* X,
                           double* result) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(result);
  CHECK_EQ(n_features, n_features_);

  // X = [n_samples, n_features]
  // coef_ = [n_features, 1]
  // ret = X . coef_ + intercept.
  sapien_set(n_samples, intercept_, result);
  sapien_gemv(internal::SAPIEN_BLAS_NO_TRANS,
              n_samples, n_features, 1.0, X, coef_.get(), 1.0, result);
}

std::vector<double> SGDRegressor::Predict(const size_t n_samples,
                                          const size_t n_features,
                                          const double* X) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(X);
  CHECK_EQ(n_features, n_features_);

  std::unique_ptr<double[]> y_pred(new double[n_samples]);
  Predict(n_samples, n_features, X, y_pred.get());
  return std::vector<double>(y_pred.get(), y_pred.get() + n_samples);
}
}  // namespace sgd
}  // namespace sapien
