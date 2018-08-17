// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Logistic Regression classifier model.

#ifndef INCLUDE_SAPIEN_LR_LOGISTIC_REGRESSION_H_
#define INCLUDE_SAPIEN_LR_LOGISTIC_REGRESSION_H_

#include <cstddef>  // size_t
#include <memory>
#include <unordered_map>

#include "sapien/lr/types.h"

namespace sapien {
namespace lr {

template<typename LabelType, typename MinimizerType = LineSearchMinimizer>
class LogisticRegression {
 public:
  struct Options {
    // Penalty strength.
    //
    // Our objective function (a.k.a cost function) is:
    //
    //  f(w) = 1/2 * w^{T} * w + C * sum(log(1 + exp(-yi * w^{T} * xi)))
    //
    // in which:
    //
    //  C = 1 / regularization_strength.
    double regularization_strength = 1e-4;

    // If fit_intercept is set to true, the weight vector and each training
    // sample is augmented as follows:
    //
    //  xi^{T} <-- [xi^{T}, 1]
    //
    //  w^{T}  <-- [w^{T}, b]
    bool fit_intercept = true;

    // Minimizer
    MinimizerType minimizer = MinimizerType();

    // The stopping criteria.
    // Training Logistic Regression simply boilds down to minimizing the
    // cost function f(w). The minimization process will stop when either
    //
    //  current_iteration_count >= max_iter
    //
    // or
    //
    //  l2_norm(f'(w)) <= epsilon * l2_norm(f'(w0)),
    //
    // in which
    //
    //  l2_norm(x) denotes the L2 norm of a vector x, and
    //  w0 is the initial value of the weight vector (i.e initial guess).
    size_t max_iter = 10;
    double epsilon = 1e-2;

    // specific training parameters

    // The weight associated with each unique class/label in the model.
    // By default it is empty meaning all weights are set to 1.0
    std::unordered_map<LabelType, double> class_weight =
        std::unordered_map<LabelType, double>();
  };

  // Construct a Logistic Regression model with default options.
  LogisticRegression();

  // Construct a Logistic Regression model with custom options.
  explicit LogisticRegression(const LogisticRegression<LabelType>::Options&
                              options);

  // We explicitly delete copy constructor and assignment operator.
  LogisticRegression(const LogisticRegression<LabelType>& src) = delete;
  LogisticRegression<LabelType>&
  operator=(const LogisticRegression<LabelType>& rhs) = delete;

  // We also explicitly delete move ctor and move assignment operator.
  LogisticRegression(LogisticRegression<LabelType>&& src) = delete;
  LogisticRegression<LabelType>&
  operator=(LogisticRegression<LabelType>&& rhs) = delete;

  // Train the model on given dataset.
  // 1. The matrix X has size of [n_samples, n_features] in row-major order
  //    meaning the first row is the first training sample, the second row
  //    is the second training sample, and so on..
  //
  // 2. The label vector y must have n_samples elements.
  //
  // Note that: if fit_intercept is set to true, each training example in
  // matrix X (along with the weight vector) will be augmented as follows:
  //
  //  xi^{T} = [xi^{T}, 1.0]
  //
  // As a consequence the matrix X will be transformed into [X, 1], i.e
  // the last column of matrix X is all ones while all previous columns are
  // the same as the old X.
  void Train(const size_t n_samples, const size_t n_features,
             const double* X, const LabelType* y);

  // Returns the mean accuracy of the predictions.
  //
  // 1. Matrix X has size of [n_samples, n_features] in row major order
  //
  //    X = [...x_i1 x_i2 .. x_iN...]  where N = n_features
  //            -----------------
  //                   ^
  //                   |
  //                   this is ith training example.
  //
  // 2. The label vector y must have n_samples elements such that
  //    y[i] is the true label associated with ith training example in
  //    matrix X.
  //
  // This is done is two steps:
  //
  // - Firstly, it computes the predicted labels for each training sample
  //   in matrix X.
  //
  // - Secondly, it computes and returns  the mean accuracy.
  //   By definition, the mean accuracy is simply the number of samples
  //   in matrix X that were correctly classified (in the first step)
  //   over the total number of samples in matrix X.
  double Score(const size_t n_samples, const size_t n_features,
               const double* X, const LabelType* y) const;

  // Predict confidence scores for samples in matrix X.
  //
  // Concrete example:
  //
  // Say our model has 2 features and 3 unique classes c0, c1, c2. The
  // coef matrix which is the result of training would look something like
  // this (assumed that fit_intercept is set to true).
  //
  //         | a00 a01 b0 | -> coefs & bias of hyperplance associated with c0
  //  coef = | a10 a11 b1 | -> .. c1
  //         | a20 a21 b2 | -> .. c2
  //
  // For each sample x^{T} = [x0, x1] in matrix X, we augment x so that
  //
  //  x^{T} = [x0, x1, 1.0]
  //
  // After that, we compute the signed distance of x to each of the above
  // hyperplanes, i.e compute the dot product of coef matrix and column
  // vector x.
  //
  //             | a00 a01 b0 |   |  x0  |
  //  coef * x = | a10 a11 b1 | * |  x1  |
  //             | a20 a21 b2 |   |  1.0 |
  //
  // If we horizontally stack all these distances for all samples in X,
  // we'll have the decision matrix. In other words,
  // decision_matrix[i * n_classes + j] is the signed distance of sample
  // xi in matrix X to the hyperplane associated with class j.
  //
  // Note that, if n_classes = 2, the decision_matrix is just a vector
  // of n_samples elements, in which decision_matrix[i] is the signed
  // distance of sample xi in matrix X to the hyperplane associated
  // with the positive class (classes_[1]). Otherwise, decision_matrix
  // has size of [n_samples, n_classes] in row-major order.
  //
  //  decision_matrix = [...d_i1 d_i2 .. d_iC ...] where C = n_classes
  //                        -----------------
  //                               ^
  //                               |
  //                               these are signed distances of sample xi
  //                               to the hyperplanes associated with each
  //                               classes_[j], j = 0, 1, .. C-1
  void DecisionMatrix(const size_t n_samples, const size_t n_features,
                      const double* X, double* decision_matrix) const;

  // Predict class labels for sample in X.
  // 1. The matrix X has size of [n_samples, n_features] in row-major order
  // 2. The predicted_labels vector must have the size of n_samples elements.
  void Predict(const size_t n_samples, const size_t n_features,
               const double* X, LabelType* predicted_labels) const;

  // Returns the 'read-only' pointer to the coef matrix.
  //
  // 1. For binary case (n_classes = 2), the pointer points to an array
  //    of n_features elements of coefficients of the predicted hyperplane
  //    saperating two classes.
  //
  // 2. For multiclass case, the pointer points to an array of
  //    n_classes * n_features elements in which:
  //    coef[i * n_features .. (i + 1) * n_features] is the coefficients
  //    of the hyperplane associated with class i.
  //
  // 3. Image:
  //
  //                     this last column consists of bias terms if
  //                     ^ fit_intercept is set to true, othwerwise
  //                     | this last column doesnot exist.
  //                     |
  //                     |
  //    | w11 w12 .. w1n b1 | -> hyperplane associated with classes[0]
  //    | w22 w21 .. w2n b2 | -> .. classes[1]
  //    | ................. | ..
  //    | ................. | ..
  //    | wC1 wC2 .. wCn bC | -> hyperplane associated with
  //      --------------         classes[n_classes - 1].
  //             ^
  //             |
  //             the number of columns here is equal to the number of
  //             features in the matrix X (i.e number of columns
  //             in matrix X)
  //
  // Note that, n_features will be equal to the actual number of features
  // plus 1 if fit_intercept is set to true.
  const double* coef() const { return coef_.get(); }

  // Returns number of features which equals to the number of features
  // in the training set (plus 1 if fit_intercept is set to true).
  size_t n_features() const { return n_features_; }

  // Returns number of classes (derived from the traing set).
  size_t n_classes() const { return n_classes_; }

  // Returns a pointer points to an array of unique classes derived
  // from training label vector.
  const LabelType* classes() const { return classes_.get(); }

 private:
  size_t n_features_;
  size_t n_classes_;

  // Unique classes derived from training data sorted in ascending order,
  // e.g. given the training label vector
  //
  //  y = [1, 1, 0, -1, -1, -1]
  //
  // classes_ would be:
  //
  //  classes_ = [-1, 0, 1]
  std::unique_ptr<LabelType[]> classes_;

  // Coeficient matrix.
  //
  // 1. Binary case:
  //    coef_ is an array of n_features_ elements representing the
  //    coeficients of the seperating hyperplane.
  //
  // 2. Multiclass case:
  //    coef_ is an array of n_features_ * n_classes_ elements in which
  //    the range coef_[i * n_features .. (i+1) * n_features] representing
  //    the coeficients of the hyperplane associated with classes_[i].
  std::unique_ptr<double[]> coef_;


  // For each training sample xi, denote:
  //
  //  sigma_i = log(1 + exp(yi * w^T * xi))
  //
  // Our cost functon become:
  //
  //  f(w) = 1/2 * w^T * w + C * sum(sigma_i)
  //  in which C = 1 / options_.regularization_strength
  //
  // weighted_C_[i] = C * class_weight_of_sample_xi
  std::unique_ptr<double[]> weighted_C_;

  // Model options
  LogisticRegression<LabelType>::Options options_;
};
}  // namespace lr
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_LR_LOGISTIC_REGRESSION_H_
