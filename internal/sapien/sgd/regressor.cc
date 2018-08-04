// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <cstring>  // memcpy

#include "sapien/sgd/regressor.h"
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
Regressor::Regressor()
    : Base(REGRESSION_MODEL, RegressorDefaultOptions()),
      summary_("Regressor: model has not been trained yet"),
      n_features_(0),
      coef_(nullptr),
      intercept_(0.0) {
}

// Constructor that takes custom options provided by user.
Regressor::Regressor(const Regressor::Options& options)
    : Base(REGRESSION_MODEL, options),
      summary_("Regressor: model has not been trained yet"),
      n_features_(0),
      coef_(nullptr),
      intercept_(0.0) {
}

// Copy ctor
Regressor::Regressor(const Regressor& src)
    : n_features_(src.n_features_),
      summary_(src.summary_),
      intercept_(src.intercept_),
      coef_(nullptr) {
  if (src.coef_ != nullptr) {
    coef_ = new double[n_features_];
    std::memcpy(coef_, src.coef_, n_features_ * sizeof(double));
  }
}

// Assignment operator
Regressor& Regressor::operator=(const Regressor& rhs) {
  if (this != &rhs) {
    size_t n_features = rhs.n_features_;
    double* coef = new double[n_features];
    std::memcpy(coef, rhs.coef_, n_features * sizeof(double));

    if (coef_ != nullptr) { delete[] coef_; }

    coef_ = coef;
    n_features_ = n_features;
    intercept_ = rhs.intercept_;
    summary_ = rhs.summary_;
  }
  return *this;
}

// Move ctor
Regressor::Regressor(Regressor&& src)
    : n_features_(src.n_features_),
      intercept_(src.intercept_),
      summary_(src.summary_),
      coef_(src.coef_) {
  src.coef_ = nullptr;
}

// Move assignment operator
Regressor& Regressor::operator=(Regressor&& rhs) {
  if (this != &rhs) {
    if (coef_ != nullptr) { delete[] coef_; }
    coef_ = rhs.coef_;
    rhs.coef_ = nullptr;

    n_features_ = rhs.n_features_;
    intercept_ = rhs.intercept_;
    summary_ = rhs.summary_;
  }
  return *this;
}

// Destructor
Regressor::~Regressor() {
  if (coef_ != nullptr) {
    delete[] coef_;
    coef_ = nullptr;
  }
}

// Train
void Regressor::Train(const size_t n_samples,
                      const size_t n_features,
                      const double* X,
                      const double* y) {
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  double start = WallTimeInSeconds();

  n_features_ = n_features;

  // Initialize coef_ vector
  // TODO(Linh): This should really be changed in order to allow client
  //             to initialize coef matrix another way.
  if (coef_ != nullptr) { delete[] coef_; }
  coef_ = new double[n_features];
  sapien_set(n_features, 0.0, coef_);

  // Initialize sample weights.
  // For the time being, all sample weights are initialized to 0.0s
  // TODO(Linh): Brainstorm of other methods to initialize sample weights.
  //             (also see Classifier).
  double* sample_weight = new double[n_samples];
  sapien_set(n_samples, 1.0, sample_weight);

  double* average_weight;
  double average_intercept = 0.0;

  const bool fit_average_sgd = this->options().average_sgd > 0;

  if (fit_average_sgd) {
    average_weight = new double[n_features];
  } else {
    average_weight = nullptr;
  }

  this->TrainOne(n_samples,
                 n_features,
                 X,
                 y,
                 sample_weight,
                 coef_,
                 &intercept_,
                 average_weight,
                 &average_intercept,
                 1.0,
                 1.0);

  if (fit_average_sgd) {
    std::memcpy(coef_, average_weight, n_features * sizeof(double));
    intercept_ = average_intercept;
    delete[] average_weight;
  }

  // Clean up.
  delete[] sample_weight;

  // Summary
  summary_ = StringPrintf("Training time: %.6f, Training score: %.6f",
                          WallTimeInSeconds() - start,
                          Score(n_samples, n_features, X, y));
}

// Score
double Regressor::Score(const size_t n_samples,
                        const size_t n_features,
                        const double* X, const double* y) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);
  CHECK_EQ(n_features, n_features_);

  double* y_pred = new double[n_samples];
  Predict(n_samples, n_features, X, y_pred);
  double score = R2Score(n_samples, y, y_pred);
  delete[] y_pred;
  return score;
}

// Predict
void Regressor::Predict(const size_t n_samples,
                        const size_t n_features,
                        const double* X,
                        double* result) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(result);
  CHECK_EQ(n_features, n_features_);

  // X = [n_samples, n_features]
  // coef_ = [n_features, 1]
  // ret = X . coef_ + intercept.
  sapien_set(n_samples, intercept_, result);
  sapien_gemv(internal::SAPIEN_BLAS_NO_TRANS,
              n_samples, n_features, 1.0, X, coef_, 1.0, result);
}

std::vector<double> Regressor::Predict(const size_t n_samples,
                                       const size_t n_features,
                                       const double* X) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(X);
  CHECK_EQ(n_features, n_features_);

  double* y_pred = new double[n_samples];
  Predict(n_samples, n_features, X, y_pred);
  std::vector<double> ret(y_pred, y_pred + n_samples);
  delete[] y_pred;
  return ret;
}
}  // namespace sgd
}  // namespace sapien
