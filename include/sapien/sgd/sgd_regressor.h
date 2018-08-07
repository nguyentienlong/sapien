// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Stochastic Gradient Descent SGDRegressor.

#ifndef INCLUDE_SAPIEN_SGD_SGD_REGRESSOR_H_
#define INCLUDE_SAPIEN_SGD_SGD_REGRESSOR_H_

#include <vector>
#include <string>
#include <memory>

#include "sapien/sgd/base.h"

namespace sapien {
namespace sgd {

class SGDRegressor : public Base {
 public:
  // Construct a SGD regressor model with default options.
  SGDRegressor();

  // Construct a SGD regressor model with custom options.
  explicit SGDRegressor(const SGDRegressor::Options& options);

  // We explicitly delet ctor and assignement operator
  SGDRegressor(const SGDRegressor& src) = delete;
  SGDRegressor& operator=(const SGDRegressor& rhs) = delete;

  // And also delete move ctor and move assignment operator
  SGDRegressor(SGDRegressor&& src) = delete;
  SGDRegressor& operator=(SGDRegressor&& rhs) = delete;

  // Train the model on given dataset.
  //
  // 1. The matrix X has n_samples rows and n_features columns in row-major
  //    order. Each row is a single training sample x.
  //
  // 2. The target vector y have n_samples elements, in which y[i] is
  //    the target associated with sample xi in X.
  void Train(const size_t n_samples, const size_t n_features,
             const double* X, const double* y);

  // Brief report after training
  std::string Summary() const { return summary_; }

  // Return the R^2 score (see [1]) of Predict(X) w.r.t y
  //
  // 1. The matrix X has n_samples rows and n_features columns in row-major
  //    order. Note that, n_features here must equal to n_features supplied
  //    to the Train method.
  //
  // 2. The ground truth target vector y has n_samples elements, in which
  //    y[i] is the true target associated with sample xi in matrix X.
  //
  // [1] - https://en.wikipedia.org/wiki/Coefficient_of_determination.
  double Score(const size_t n_samples, const size_t n_features,
               const double* X, const double* y) const;

  // Predict target values for samples in X.
  // Upon returned successfully, result[i] will be the predicted target
  // for sample xi in matrix X.
  void Predict(const size_t n_samples, const size_t n_features,
               const double* X, double* result) const;

  // Predict targt values for samples in X.
  std::vector<double> Predict(const size_t n_samples,
                              const size_t n_features,
                              const double* X) const;

  // Returns the number of features.
  size_t n_features() const { return n_features_; }

  // Returns the coefficients vector
  const double* coef() const { return coef_.get(); }

  // Return the intercept value.
  double intercept() const { return intercept_; }

 private:
  size_t n_features_;
  std::unique_ptr<double[]> coef_;
  double intercept_;
  std::string summary_;
};
}  // namespace sgd
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SGD_SGD_REGRESSOR_H_
