// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include "sapien/lr/logistic_regression.h"
#include "sapien/metrics.h"
#include "sapien/internal/sapien_math.h"
#include "sapien/utility/unique.h"
#include "glog/logging.h"

namespace sapien {
namespace lr {

using internal::AccuracyScore;
using internal::UniqueElements;

using internal::sapien_set;

// Construct a Logistic Regression model with default options
template<typename LabelType>
LogisticRegression<LabelType>::LogisticRegression()
    : options_(LogisticRegression<LabelType>::Options()) {}

// Construct a Logistic Regression model with custom options.
template<typename LabelType>
LogisticRegression<LabelType>::
LogisticRegression(const LogisticRegression<LabelType>::Options& options)
    : options_(options) {}

// Returns the mean accuracy of the predictions.
template<typename LabelType>
double LogisticRegression<LabelType>::
Score(const size_t n_samples,
      const size_t n_features,
      const double* X,
      const LabelType* y) const {
  // Make sure that our model has already been fitted
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(classes_.get());

  // CHECK input parameters
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  const size_t n_augmented_features = (options_.fit_intercept) ?
      (n_features + 1) : n_features;
  CHECK_EQ(n_augmented_features, n_features_);

  std::unique_ptr<LabelType> predicted_labels(new LabelType[n_samples]);
  Predict(n_samples, n_features, X, predicted_labels.get());
  return AccuracyScore(n_samples, y, predicted_labels.get());
}

// Decision matrix.
template<typename LabelType>
void LogisticRegression<LabelType>::
DecisionMatrix(const size_t n_samples,
               const size_t n_features,
               const double* X,
               double* decision_matrix) const {
  // Make sure that the model has already been fitted/trained.
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(classes_.get());

  // CHECK input parameters
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(decision_matrix);

  const size_t n_augmented_features = (options_.fit_intercept) ?
      (n_features + 1) : n_features;
  CHECK_EQ(n_augmented_features, n_features_);

  // Number of seperating hyperplanes
  const size_t n_planes = (n_classes_ == 2) ? 1 : n_classes_;

  if (!options_.fit_intercept) {  // no augmentations needed
    // We have:
    //  decision_matrix = X * coef_^T
    // or in BLAS term:
    //  decision_matrix <- 1.0 * X * coef_^T + 0.0 * ret;
    sapien_gemm(internal::SAPIEN_BLAS_NO_TRANS,
                internal::SAPIEN_BLAS_TRANS,
                n_samples, n_planes, n_features, 1.0,
                X, coef_.get(), 0.0, decision_matrix);
  } else {
    // First we need to augment matrix X by appending column of all 1s to X.
    // Note that the size of the aumented matrix is
    // [n_samples, n_features_]
    //
    // TODO(Linh): This is quite expensive. Do we really to augment X?

    std::unique_ptr<double[]> augmented_X(new double[n_samples * n_features_]);
    for (size_t i = 0; i < n_samples; ++i) {
      for (size_t j = 0; j < n_features_; ++j) {
        if (j == n_features) {
          augmented_X[i * n_features_ + j] = 1.0;
        } else {
          augmented_X[i * n_features_ + j] = X[i * n_features + j];
        }
      }
    }

    // Then decision_matrix is simply the dot product of the augmented
    // matrix and the tranpose of coef_
    sapien_gemm(internal::SAPIEN_BLAS_NO_TRANS,
                internal::SAPIEN_BLAS_TRANS,
                n_samples, n_planes, n_features_, 1.0,
                X, coef_.get(), 0.0, decision_matrix);
  }
}

// Predict the labels for samples in X.
template<typename LabelType>
void LogisticRegression<LabelType>::
Predict(const size_t n_samples, const size_t n_features,
        const double* X, LabelType* predicted_labels) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(classes_.get());

  CHECK_NOTNULL(X);
  CHECK_NOTNULL(predicted_labels);

  const size_t n_augmented_features = (options_.fit_intercept) ?
      (n_features + 1) : n_features;
  CHECK_EQ(n_augmented_features, n_features_);

  const size_t n_planes = (n_classes_ == 2) ? 1 : n_classes_;

  // Decision matrix (a.k.a decision function)
  std::unique_ptr<double[]>
      decision_matrix(new double[n_samples * n_planes]);
  DecisionMatrix(n_samples, n_features, X, decision_matrix.get());

  if (n_classes_ == 2) {
    // For binary case, we predict classes_[1] for samples whose signed
    // distance to the hyperplane is positive, otherwise classes_[0].
    predicted_labels[i] = (decision_matrix[i] > 0) ? classes_[1] : classes_[0];
  } else {
    // For multiclass case, we predict the classes_[imax] for each sample
    // in matrix X, where imax is the index of the maximum distance
    // among all distances of x to all hyperplanes.
    for (size_t i = 0; i < n_samples; ++i) {
      size_t imax = sapien_imax(n_planes,
                                decision_matrix.get() + i * n_planes);
      predicted_labels[i] = classes_[imax];
    }
  }
}

template<typename LabelType>
void LogisticRegression<LabelType>::Train(const size_t n_samples,
                                          const size_t n_features,
                                          const double* X,
                                          const LabelType* y) {
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  // Extract unique labels from the input label vector y into classes_.
  // For example:
  //  with y = [1, 1, -1, -1, -1, 0, 1, 0], we have:
  //  classes_ = [-1, 0, 1] & n_classes_ = 3
  classes_.reset(UniqueElements(n_samples, y, &n_classes_));
  CHECK_GT(n_classes_, 1);

  n_features_ = (options_.fit_intercept) ? (n_features + 1) : n_features;
  const size_t n_planes = (n_classes_ == 2) ? 1 : n_classes_;

  // Setup the coef_ matrix
  coef_.reset(new double[n_planes * n_features_]);
  sapien_set(n_planes * n_features_, 0.0, coef_.get());

  // Setup weighted_C_

  const double C = 1.0 / options_.regularization_strength;
  weighted_C_.reset(new double[n_classes_]);
  sapien_set(n_classes_, C, weighted_C_.get());

  if (!options_.class_weight.empty()) {
    for (size_t i = 0; i < n_classes_; ++i) {
      const LabelType label = classes_[i];
      std::unordered_map<LabelType, double>::iterator it;
      it = options_.class_weight.find(label);
      if (it != options_.class_weight.end()) {
        weighted_C_[i] *= it->second;
      }
    }
  }

  // One vs Rest

  if (n_classes_ == 2) {
    // We treat classes_[1] as the positive class, and classes_[0] as
    // negative
    OneVsRest(n_samples, n_features, X, y, 1);
  } else {
    // We treat classes_[c] as the positive class, and the rest is negative
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      OneVsRest(n_samples, n_features, X, y, class_idx);
    }
  }
}

// One vs Rest
template<typename LabelType>
void LogisticRegression<LabelType>::OneVsRest(const size_t n_samples,
                                              const size_t n_features,
                                              const double* X,
                                              const LabelType* y,
                                              const size_t class_idx) {
  // Extract labels for binary training from y
  std::unique_ptr<double[]> targets(new double[n_samples]);
  const LabelType positive_label = classes_[class_idx];
  LabelType label, diff;
  for (size_t i = 0; i < n_samples; ++i) {
    label = y[i];
    diff = label - positive_class;
    if (diff == LabelType(0)) {
      labels[i] = 1.0;
    } else {
      labels[i] = -1.0;
    }
  }

  // Exract weighted C
  double pos_C, neg_C;
  if (n_classes_ == 2) {
    pos_C = weighted_C_[1];
    neg_C = weighted_C_[0];
  } else {
    pos_C = weighted_C_[class_idx];
    neg_C = 1 / options_.regularization_strength;
  }

  // Establish the cost function
  LogisticRegressionCostFunction cost_function(n_samples,
                                               n_features,
                                               X,
                                               y,
                                               pos_C,
                                               neg_C);

  // And the minimizer
  minimizer.Minimizer(const_function, weight);
}

}  // namespace lr
}  // namespace sapien
