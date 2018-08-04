// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Stochastic Gradient Descent classifier.

#ifndef INCLUDE_SAPIEN_SGD_CLASSIFIER_H_
#define INCLUDE_SAPIEN_SGD_CLASSIFIER_H_

#include <cstddef>  // size_t
#include <string>
#include <vector>

#include "sapien/sgd/base.h"

namespace sapien {
namespace sgd {

template<typename LabelType>
class Classifier : public Base {
 public:
  // Construct a SGD classifier model with default options.
  Classifier();

  // Construct a SGD classifier model with custom options.
  explicit Classifier(const Classifier::Options& options);

  // Copy constructor and assignment operator
  Classifier(const Classifier& src);
  Classifier& operator=(const Classifier& rhs);

  // Move constructor & move assignment operator
  Classifier(Classifier&& src);
  Classifier& operator=(Classifier&& rhs);

  // Destructor
  ~Classifier();

  // Train the model on given dataset.
  // 1. The matrix X has size of [n_samples, n_features] in row-major order.
  // 2. The label vector y must have the size of n_samples.
  // TODO(Linh): How about to add another additional parameters to allow
  // user to, say choose a particular initial values for the coefficients.
  void Train(const size_t n_samples, const size_t n_features,
             const double* X, const LabelType* y);

  // Brief report after training.
  std::string Summary() const { return summary_; }

  // Return the mean accuracy of Predict(X) w.r.t y.
  // 1. The matrix X has the size of [n_samples, n_features] in row-major
  //    order.
  // 2. The label vector must have n_samples elements.
  double Score(const size_t n_samples, const size_t n_features,
               const double* X, const LabelType* y) const;

  // Predict confidence scores for samples.
  // Concrete example:
  // Say our model has 2 features and 3 unique classes
  // c0, c1, c2. The coef matrix which is the result of the training
  // would look something like this:
  //
  //         | a00 a01 | --> coefs of hyperplane associated with class c0
  // coef =  | a10 a11 | --> coefs of hyperplane associated with class c1
  //         | a20 a21 | --> coefs of hyperplane associated with class c2
  //
  // And the intercept/bias vector would look something like this:
  //
  //              | b0 | --> bias of hyperplabe associated with class c0
  // intercept =  | b1 | --> .. c1
  //              | b2 | --> .. c2
  //
  // For each sample x = [x0, x1] in matrix X, we compute the distances
  // of x to each of the above three hyperplanes, i.e to compute:
  // coef * intercept (dot product between a matrix coef and column vector
  //                  intercept)
  // If we horizontally stack all these numbers for all samples in X, we
  // would have the decision matrix.
  //
  // Upon successfully returned, ret[i * n_classes + j] is the distance
  // of sample xi in matrix X to the hyperplane associated with class j.
  void DecisionMatrix(const size_t n_samples, const size_t n_features,
                      const double* X, double* ret) const;


  // Predict class labels for sample in X.
  //
  // 1. Matrix X has n_samples row and n_features column in row-major order.
  // 2. The result array ret must have n_samples elements.
  // 3. Upon sucessfully returned, ret[i] is the predicted label for sample
  //    xi in matrix X.
  void Predict(const size_t n_samples, const size_t n_features,
               const double* X, LabelType* ret) const;

  // Predict class labels for sample in X.
  std::vector<LabelType> Predict(const size_t n_samples,
                                 const size_t n_features,
                                 const double* X) const;

  // Returns the 'read-only' pointer to the coef matrix.
  //
  // 1. For binary case, the pointer points to an array of n_features
  //    elements of coefficients of the predicted hyperplane that
  //    seperating two classes.
  //
  // 2. For multiclass case, the pointer points to an array of
  //   n_classes * n_features elements in which:
  //   coef[i * n_features .. (i + 1) * n_features] is the coefficients of
  //   the hyperplane associated with class i.
  const double* coef() const { return coef_; }

  // Return the 'read-only' poniter to the intercept vector.
  //
  // 1. For binary case, the pointer points to an array of single elment,
  //    which is the intercept/bias of the seperating hyperplane.
  //
  // 2. For multiclass case, the pointer points to an array of n_classes
  //    elements in which intercept[i] is the intercept/bias of the
  //    hyperplane associated with class i.
  const double* intercept() const { return intercept_; }

  // Returns number of features.
  size_t n_features() const { return n_features_; }

  // Returns number of classes.
  size_t n_classes() const { return n_classes_; }

 private:
  // Number of features.
  size_t n_features_;

  // Number of classes
  size_t n_classes_;

  // Coeficient matrix (row-major) order. Row i is the coefficients of the
  // hyperplane asscociated with classes_[i], and the size of each row is
  // n_features_.
  // If the model is trained with a binary dataset, the size of coef_ is
  // n_features_, and it stores the coefficients of the hyperplabe associated
  // with the positive class (classes_[1]).
  // If the model us trained with a multiclass dataset, the size of coef_ is
  // n_features_ * n_classes_.
  // [n_classes_, n_features_] or [1, n_features]
  double* coef_;

  // intercept vector. The size of intercept_ is determined as follow:
  //     sizeof(intercept_) = (n_classes_ == 2) ? 1 : n_classes_;
  double* intercept_;

  // Unique classes derived from label vector y, all sorted in ascending
  // order.
  LabelType* classes_;

  // Summary after training
  std::string summary_;

 private:
  // One vs rest
  void OneVsRest_(const size_t n_samples,
                  const size_t n_features,
                  const double* X,
                  const LabelType* y,
                  const double* sample_weight,
                  const double* class_weight,
                  const size_t c);
};
}  // namespace sgd
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SGD_CLASSIFIER_H_
