// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <algorithm>
#include <cstring>  // memcpy

#include "sapien/sgd/classifier.h"
#include "sapien/sgd/default_options.h"
#include "sapien/metrics.h"
#include "sapien/internal/sapien_math.h"
#include "sapien/utility/wall_time.h"
#include "sapien/utility/stringprintf.h"
#include "glog/logging.h"

namespace sapien {
namespace sgd {

using internal::sapien_gemm;
using internal::sapien_imax;
using internal::sapien_set;
using metrics::AccuracyScore;

using internal::StringPrintf;
using internal::WallTimeInSeconds;

// Default constructor
template<typename LabelType>
Classifier<LabelType>::Classifier()
    : Base(CLASSIFICATION_MODEL, ClassifierDefaultOptions()),
      summary_("Classifier: model has not been trained yet."),
      n_features_(0),
      n_classes_(0),
      coef_(nullptr),
      intercept_(nullptr),
      classes_(nullptr) {
}

// Constructor that takes custom options provided by user.
template<typename LabelType>
Classifier<LabelType>::Classifier(const Classifier<LabelType>::Options& options)
    : Base(CLASSIFICATION_MODEL, options),
      summary_("Classifier: model has not been trained yet."),
      n_features_(0),
      n_classes_(0),
      coef_(nullptr),
      intercept_(nullptr),
      classes_(nullptr) {
}

// Destructor
template<typename LabelType>
Classifier<LabelType>::~Classifier() {
  if (coef_ != nullptr) { delete[] coef_; }
  if (intercept_ != nullptr) { delete[] intercept_; }
  if (classes_ != nullptr) { delete[] classes_; }

  coef_ = nullptr;
  intercept_ = nullptr;
  classes_ = nullptr;
}

// Copy ctor
template<typename LabelType>
Classifier<LabelType>::Classifier(const Classifier<LabelType>& src)
    : n_features_(src.n_features_),
      n_classes_(src.n_classes_),
      summary_(src.summary_),
      coef_(nullptr),
      intercept_(nullptr),
      classes_(nullptr) {
  if (src.coef_ != nullptr) {
    size_t coef_size = ((n_classes_ == 2) ? 1 : n_classes_) * n_features_;
    coef_ = new double[coef_size];
    std::memcpy(coef_, src.coef_, sizeof(double) * coef_size);
  }

  if (src.intercept_ != nullptr) {
    size_t intercept_size = (n_classes_ == 2) ? 1 : n_classes_;
    intercept_ = new double[intercept_size];
    std::memcpy(intercept_, src.intercept_, sizeof(double) * intercept_size);
  }

  if (src.classes_ != nullptr) {
    classes_ = new LabelType[n_classes_];
    std::memcpy(classes_, src.classes_, sizeof(LabelType) * n_classes_);
  }
}

// Assignment operator
template<typename LabelType>
Classifier<LabelType>&
Classifier<LabelType>::operator=(const Classifier<LabelType>& rhs) {
  if (this != &rhs) {
    size_t intercept_size = (rhs.n_classes_ == 2) ? 1 : rhs.n_classes_;
    size_t coef_size = intercept_size * rhs.n_features_;

    double* coef = new double[coef_size];
    double* intercept = new double[intercept_size];
    LabelType* classes = new LabelType[rhs.n_classes_];

    std::memcpy(coef, rhs.coef_, sizeof(double) * coef_size);
    std::memcpy(intercept, rhs.intercept_, sizeof(double) * intercept_size);
    std::memcpy(classes, rhs.classes_, sizeof(LabelType) * rhs.n_classes_);

    if (coef_ != nullptr) { delete[] coef_; }
    if (intercept_ != nullptr) { delete[] intercept_; }
    if (classes_ != nullptr) { delete[] classes_; }

    coef_ = coef;
    intercept_ = intercept;
    classes_ = classes;

    n_features_ = rhs.n_features_;
    n_classes_ = rhs.n_classes_;
    summary_ = rhs.summary_;
  }
  return *this;
}

// Move ctor.
template<typename LabelType>
Classifier<LabelType>::Classifier(Classifier<LabelType>&& src)
    : n_features_(src.n_features_),
      n_classes_(src.n_classes_),
      summary_(src.summary_),
      coef_(src.coef_),
      intercept_(src.intercept_),
      classes_(src.classes_) {
  src.coef_ = nullptr;
  src.intercept_ = nullptr;
  src.classes_ = nullptr;
}

// Move assignment operator
template<typename LabelType>
Classifier<LabelType>&
Classifier<LabelType>::operator=(Classifier<LabelType>&& rhs) {
  if (this != &rhs) {
    if (coef_ != nullptr) { delete[] coef_; }
    if (intercept_ != nullptr) { delete[] intercept_; }
    if (classes_ != nullptr) { delete[] classes_; }

    coef_ = rhs.coef_;
    intercept_ = rhs.intercept_;
    classes_ = rhs.classes_;

    rhs.coef_ = nullptr;
    rhs.intercept_ = nullptr;
    rhs.classes_ = nullptr;

    n_features_ = rhs.n_features_;
    n_classes_ = rhs.n_classes_;
    summary_ = rhs.summary_;
  }
  return *this;
}

// Train
template<typename LabelType>
void
Classifier<LabelType>::Train(const size_t n_samples,
                             const size_t n_features,
                             const double* X,
                             const LabelType* y) {
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  double start = WallTimeInSeconds();

  // We need to check whether the custom options provided
  // are indeed valid.
  // TODO(Linh): Consider to hand this task to constructor to handle,
  //             because this should be done at the point of constructing
  //             Classifier object with custom options!
  std::string error;
  bool valid = this->options().IsValid(CLASSIFICATION_MODEL, &error);
  CHECK(valid) << error;

  n_features_ = n_features;

  // Find unique classes from y and sort them all in ascending order.
  // Let's say y = [1, 1, -1, -1, 2], the output would be [-1, 1, 2].

  LabelType* copied_y = new LabelType[n_samples];
  std::memcpy(copied_y, y, n_samples * sizeof(LabelType));
  std::sort(copied_y, copied_y + n_samples);

  size_t n_uniques = 1;

  for (size_t i = 1; i < n_samples; ++i) {
    const LabelType a = copied_y[i-1];
    const LabelType b = copied_y[i];
    const LabelType diff = a - b;
    if (diff != LabelType(0)) { ++n_uniques; }
  }

  // Make sure that we have at least two classes.
  CHECK_GT(n_uniques, 1);

  n_classes_ = n_uniques;
  if (classes_ != nullptr) { delete[] classes_; }
  classes_ = new LabelType[n_uniques];

  LabelType* ptr = classes_;
  *ptr = *copied_y;
  ptr++;

  for (size_t i = 1; i < n_samples; ++i) {
    const LabelType a = copied_y[i-1];
    const LabelType b = copied_y[i];
    const LabelType diff = a - b;
    if (diff != LabelType(0)) {
      *ptr = b;
      ptr++;
    }
  }

  // Setup for training.

  size_t intercept_size = (n_classes_ == 2) ? 1 : n_classes_;

  // Initialize coef_ matrix. For now all coefficients are initialized to 0.0
  // TODO(Linh)
  if (coef_ != nullptr) { delete[] coef_; }
  size_t coef_size = intercept_size * n_features_;
  coef_ = new double[coef_size];
  sapien_set(coef_size, 0.0, coef_);

  // Initialize intercept_ vector. For the moment, all intercepts/biases
  // are initialized to 0.0
  // TODO(Linh)
  if (intercept_ != nullptr) { delete[] intercept_; }
  intercept_ = new double[intercept_size];
  sapien_set(intercept_size, 0.0, intercept_);

  // Sample weight. For the moment, all sample weights are initilized to 1.0
  // TODO(Linh)
  double* sample_weight = new double[n_samples];
  sapien_set(n_samples, 1.0, sample_weight);

  // Class weight.  For now, just set all the weights to 1.0.
  // TODO(Linh)
  double* class_weight = new double[n_classes_];
  sapien_set(n_classes_, 1.0, class_weight);

  if (n_classes_ == 2) {
    OneVsRest_(n_samples, n_features, X, y, sample_weight, class_weight, 1);
  } else {
    for (size_t c = 0; c < n_classes_; ++c) {
      OneVsRest_(n_samples, n_features, X, y, sample_weight, class_weight, c);
    }
  }

  // Cleanup
  delete[] sample_weight;
  delete[] class_weight;

  summary_ = StringPrintf("Training time: %.6fs, Traing score: %.6f",
                          WallTimeInSeconds() - start,
                          Score(n_samples, n_features, X, y));
}

// Returns mean accuracy score.
template<typename LabelType>
double
Classifier<LabelType>::Score(const size_t n_samples,
                             const size_t n_features,
                             const double* X,
                             const LabelType* y) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(intercept_);
  CHECK_NOTNULL(classes_);

  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  LabelType* y_pred = new LabelType[n_samples];
  Predict(n_samples, n_features, X, y_pred);
  double score =  AccuracyScore(n_samples, y, y_pred);
  delete[] y_pred;
  return score;
}

// Decision function
template<typename LabelType>
void
Classifier<LabelType>::DecisionMatrix(const size_t n_samples,
                                      const size_t n_features,
                                      const double* X,
                                      double* ret) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(intercept_);
  CHECK_NOTNULL(classes_);

  CHECK_NOTNULL(X);
  CHECK_NOTNULL(ret);
  CHECK_EQ(n_features, n_features_);

  size_t intercept_size = (n_classes_ == 2) ? 1 : n_classes_;

  // Populate biases/intercepts to ret.
  // ret is expected to have the size of [n_samples, intercept_size]
  // We now populate each row of ret with the intercept_ vector.
  double* ret_rowi;
  for (size_t i = 0; i < n_samples; ++i) {  // each row
    ret_rowi = ret + i * intercept_size;
    std::memcpy(ret_rowi, intercept_, sizeof(double) * intercept_size);
  }

  // ret <- 1.0 * X . coef_^T + 1.0 * ret.
  //
  // X = [n_samples, n_features]
  // coef_ = [intercept_size, n_features]
  // X * coef_.t() = [n_samples, intercept_size]
  // ret = [n_samples, intercept_size]
  sapien_gemm(internal::SAPIEN_BLAS_NO_TRANS,
              internal::SAPIEN_BLAS_TRANS,
              n_samples, intercept_size, n_features, 1.0,
              X, coef_, 1.0, ret);
}

// Predict the labels for samples in X.
template<typename LabelType>
void
Classifier<LabelType>::Predict(const size_t n_samples,
                               const size_t n_features,
                               const double* X,
                               LabelType* ret) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(intercept_);
  CHECK_NOTNULL(classes_);
  
  CHECK_EQ(n_features, n_features_);
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(ret);

  // TODO(Linh): This line of code is so repetitive, consider
  // to factor this out!
  size_t intercept_size = (n_classes_ == 2) ? 1 : n_classes_;

  double* decision = new double[n_samples * intercept_size];
  DecisionMatrix(n_samples, n_features, X, decision);

  if (n_classes_ == 2) {
    for (size_t i = 0; i < n_samples; ++i) {
      // For the binary case, we predict classes_[1] for samples whose
      // distance to the hyperplane is positive, othersie classes_[0]
      ret[i] = (decision[i] > 0) ? classes_[1] : classes_[0];
    }
  } else {
    // For multiclass case, we predict the classes_[imax] for each sample
    // x in matrix X, where imax is the index of the maximum distance among
    // all distances of x to all hyperplanes.
    for (size_t i = 0; i < n_samples; ++i) {
      size_t imax = sapien_imax(intercept_size, decision + i*intercept_size);
      ret[i] = classes_[imax];
    }
  }

  delete[] decision;
}

// Predict class labels for samples in X.
template<typename LabelType>
std::vector<LabelType>
Classifier<LabelType>::Predict(const size_t n_samples,
                               const size_t n_features,
                               const double* X) const {
  CHECK_NOTNULL(coef_);
  CHECK_NOTNULL(intercept_);
  CHECK_NOTNULL(classes_);

  CHECK_NOTNULL(X);
  LabelType* predicted_labels = new LabelType[n_samples];
  Predict(n_samples, n_features, X, predicted_labels);
  std::vector<LabelType> ret(predicted_labels, predicted_labels + n_samples);
  delete[] predicted_labels;
  return ret;
}

// One vs Rest
template<typename LabelType>
void
Classifier<LabelType>::OneVsRest_(const size_t n_samples,
                                  const size_t n_features,
                                  const double* X,
                                  const LabelType* y,
                                  const double* sample_weight,
                                  const double* class_weight,
                                  const size_t c) {
  // Extract the labels for binary training from y.
  double* labels = new double[n_samples];
  const LabelType pos_cls = classes_[c];
  for (size_t i = 0; i < n_samples; ++i) {
    const LabelType cls = y[i];
    const LabelType diff = cls - pos_cls;
    if (diff == LabelType(0)) {
      labels[i] = 1.0;
    } else {
      labels[i] = -1.0;
    }
  }

  // Extract class weight.
  double weight_pos, weight_neg;
  if (n_classes_ == 2) {
    weight_pos = class_weight[1];
    weight_neg = class_weight[0];
  } else {
    weight_pos = class_weight[c];
    weight_neg = 1.0;
  }

  // Prepare storage to store the outputs of training.
  const size_t idx = (n_classes_ == 2) ? 0 : c;
  double* weight = coef_ + idx * n_features;
  double* intercept = intercept_ + idx;

  double* average_weight;
  double average_intercept = 0.0;

  const bool average_sgd = (this->options().average_sgd > 0);

  if (average_sgd) {
    average_weight = new double[n_features];
    sapien_set(n_features, 0.0, average_weight);
  } else {
    average_weight = nullptr;
  }

  // Training
  this->TrainOne(n_samples,
                 n_features,
                 X,
                 labels,
                 sample_weight,
                 weight,
                 intercept,
                 average_weight,
                 &average_intercept,
                 weight_pos,
                 weight_neg);
  // ASGD?
  if (average_sgd) {
    std::memcpy(weight, average_weight, n_features * sizeof(double));
    *intercept = average_intercept;
    delete[] average_weight;
  }

  delete[] labels;
}

template class Classifier<int>;
template class Classifier<uint8_t>;
}  // namespace sgd
}  // namespace sapien

