// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <algorithm>
#include <cstring>  // memcpy

#include "sapien/internal/port.h"
#include "sapien/sgd/sgd_classifier.h"
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
SGDClassifier<LabelType>::SGDClassifier()
    : Base(ClassifierDefaultOptions()),
      summary_("SGDClassifier: model has not been trained yet."),
      n_features_(0),
      n_classes_(0),
      coef_(nullptr),
      intercept_(nullptr),
      classes_(nullptr) {
  // We use Hinge loss as the default loss functor
  Base::loss_functor(new HingeLoss<double>(1.0));
}

// Constructor that takes custom options provided by user.
template<typename LabelType>
SGDClassifier<LabelType>::
SGDClassifier(const SGDClassifier<LabelType>::Options& options)
    : Base(options),
      summary_("SGDClassifier: model has not been trained yet."),
      n_features_(0),
      n_classes_(0),
      coef_(nullptr),
      intercept_(nullptr),
      classes_(nullptr) {
  // We use Hinge loss as the default loss functor
  Base::loss_functor(new HingeLoss<double>(1.0));
}

// Train
template<typename LabelType>
void
SGDClassifier<LabelType>::Train(const size_t n_samples,
                                const size_t n_features,
                                const double* X,
                                const LabelType* y) {
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  double start = WallTimeInSeconds();

  // We need to check whether the custom options provided
  // are indeed valid.
  std::string error;
  bool valid = this->options().IsValid(&error);
  CHECK(valid) << error;

  // We also need to check to make sure that our loss functor is not
  // NULL and it is indeed used for classification model.
  CHECK_NOTNULL(this->loss_functor());
  CHECK(this->loss_functor()->IsClassification())
      << "Invalid loss functor.";

  n_features_ = n_features;

  // Find unique classes from y and sort them all in ascending order.
  // Let's say y = [1, 1, -1, -1, 2], the output would be [-1, 1, 2].

  std::unique_ptr<LabelType[]> copied_y(new LabelType[n_samples]);
  std::memcpy(copied_y.get(), y, n_samples * sizeof(LabelType));
  std::sort(copied_y.get(), copied_y.get() + n_samples);

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
  classes_.reset(new LabelType[n_uniques]);

  LabelType* ptr = classes_.get();
  *ptr = copied_y[0];
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
  size_t coef_size = intercept_size * n_features_;
  coef_.reset(new double[coef_size]);
  sapien_set(coef_size, 0.0, coef_.get());

  // Initialize intercept_ vector. For the moment, all intercepts/biases
  // are initialized to 0.0
  // TODO(Linh)
  intercept_.reset(new double[intercept_size]);
  sapien_set(intercept_size, 0.0, intercept_.get());

  // Sample weight. For the moment, all sample weights are initilized to 1.0
  // TODO(Linh)
  std::unique_ptr<double[]> sample_weight(new double[n_samples]);
  sapien_set(n_samples, 1.0, sample_weight.get());

  // Class weight.  For now, just set all the weights to 1.0.
  // TODO(Linh)
  std::unique_ptr<double[]> class_weight(new double[n_classes_]);
  sapien_set(n_classes_, 1.0, class_weight.get());

  if (n_classes_ == 2) {
    OneVsRest(n_samples, n_features, X, y, sample_weight.get(),
              class_weight.get(), 1);
  } else {
    for (size_t c = 0; c < n_classes_; ++c) {
      OneVsRest(n_samples, n_features, X, y, sample_weight.get(),
                class_weight.get(), c);
    }
  }

  summary_ = StringPrintf("Training time: %.6fs, Traing score: %.6f",
                          WallTimeInSeconds() - start,
                          Score(n_samples, n_features, X, y));
}

// Returns mean accuracy score.
template<typename LabelType>
double
SGDClassifier<LabelType>::Score(const size_t n_samples,
                                const size_t n_features,
                                const double* X,
                                const LabelType* y) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(intercept_.get());
  CHECK_NOTNULL(classes_.get());

  CHECK_NOTNULL(X);
  CHECK_NOTNULL(y);

  std::unique_ptr<LabelType[]> y_pred(new LabelType[n_samples]);
  Predict(n_samples, n_features, X, y_pred.get());
  return AccuracyScore(n_samples, y, y_pred.get());
}

// Decision function
template<typename LabelType>
void
SGDClassifier<LabelType>::DecisionMatrix(const size_t n_samples,
                                         const size_t n_features,
                                         const double* X,
                                         double* ret) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(intercept_.get());
  CHECK_NOTNULL(classes_.get());

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
    std::memcpy(ret_rowi, intercept_.get(), sizeof(double) * intercept_size);
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
              X, coef_.get(), 1.0, ret);
}

// Predict the labels for samples in X.
template<typename LabelType>
void
SGDClassifier<LabelType>::Predict(const size_t n_samples,
                                  const size_t n_features,
                                  const double* X,
                                  LabelType* ret) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(intercept_.get());
  CHECK_NOTNULL(classes_.get());

  CHECK_EQ(n_features, n_features_);
  CHECK_NOTNULL(X);
  CHECK_NOTNULL(ret);

  // TODO(Linh): This line of code is so repetitive, consider
  // to factor this out!
  size_t intercept_size = (n_classes_ == 2) ? 1 : n_classes_;

  std::unique_ptr<double[]>
      decision_matrix(new double[n_samples * intercept_size]);
  DecisionMatrix(n_samples, n_features, X, decision_matrix.get());

  if (n_classes_ == 2) {
    for (size_t i = 0; i < n_samples; ++i) {
      // For the binary case, we predict classes_[1] for samples whose
      // distance to the hyperplane is positive, othersie classes_[0]
      ret[i] = (decision_matrix[i] > 0) ? classes_[1] : classes_[0];
    }
  } else {
    // For multiclass case, we predict the classes_[imax] for each sample
    // x in matrix X, where imax is the index of the maximum distance among
    // all distances of x to all hyperplanes.
    for (size_t i = 0; i < n_samples; ++i) {
      size_t imax = sapien_imax(intercept_size,
                                decision_matrix.get() + i*intercept_size);
      ret[i] = classes_[imax];
    }
  }
}

// Predict class labels for samples in X.
template<typename LabelType>
std::vector<LabelType>
SGDClassifier<LabelType>::Predict(const size_t n_samples,
                                  const size_t n_features,
                                  const double* X) const {
  CHECK_NOTNULL(coef_.get());
  CHECK_NOTNULL(intercept_.get());
  CHECK_NOTNULL(classes_.get());

  CHECK_NOTNULL(X);

  std::unique_ptr<LabelType[]> predicted_labels(new LabelType[n_samples]);
  Predict(n_samples, n_features, X, predicted_labels.get());
  return std::vector<LabelType>(predicted_labels.get(),
                                predicted_labels.get() + n_samples);
}

// One vs Rest
template<typename LabelType>
void
SGDClassifier<LabelType>::OneVsRest(const size_t n_samples,
                                    const size_t n_features,
                                    const double* X,
                                    const LabelType* y,
                                    const double* sample_weight,
                                    const double* class_weight,
                                    const size_t c) {
  // Extract the labels for binary training from y.
  std::unique_ptr<double[]> labels(new double[n_samples]);
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
  double* weight = coef_.get() + idx * n_features;
  double* intercept = intercept_.get() + idx;

  std::unique_ptr<double[]> average_weight;
  double average_intercept = 0.0;

  const bool average_sgd = (this->options().average_sgd > 0);

  if (average_sgd) {
    average_weight = std::unique_ptr<double[]>(new double[n_features]);
    sapien_set(n_features, 0.0, average_weight.get());
  }

  // Training
  this->TrainOne(n_samples,
                 n_features,
                 X,
                 labels.get(),
                 sample_weight,
                 weight,
                 intercept,
                 average_weight.get(),
                 &average_intercept,
                 weight_pos,
                 weight_neg);
  // ASGD?
  if (average_sgd) {
    std::memcpy(weight, average_weight.get(), n_features * sizeof(double));
    *intercept = average_intercept;
  }
}

// We instantiate SGDClassifier class template for most common
// integral types.
template class SGDClassifier<uint8_t>;
template class SGDClassifier<int8_t>;
template class SGDClassifier<uint16_t>;
template class SGDClassifier<int16_t>;
template class SGDClassifier<uint32_t>;
template class SGDClassifier<int32_t>;
template class SGDClassifier<uint64_t>;
template class SGDClassifier<int64_t>;
}  // namespace sgd
}  // namespace sapien

