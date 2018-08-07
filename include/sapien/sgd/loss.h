// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// LossFunctor interface and some common concrete loss functors (which
// implements LossFunctor interface) used by Stochastic Gradient Descent
// algorithm.
//
// User could either choose to use one of these concrete loss functors
// (recommended) or provide their own loss functor by directly implementing
// LostFunctor interface when constructing SGD model.

#ifndef INCLUDE_SAPIEN_SGD_LOSS_H_
#define INCLUDE_SAPIEN_SGD_LOSS_H_

namespace sapien {
namespace sgd {

// LossFunctor interface. The type T could be either float or double.
template<typename T>
struct LossFunctor {
  // Returns true if this loss functor is used for classification model,
  // false if it is used for regression model.
  virtual bool IsClassification() const = 0;

  // Evaluate the loss at the prediction p w.r.t the ground truth y.
  virtual T operator()(const T p, const T y) const = 0;

  // Evaluate the first derivative of the loss function w.r.t the prediction
  // p.
  virtual T FirstDerivative(const T p, const T y) const = 0;

  virtual ~LossFunctor() {}
};

// Concrete loss functor which implements LossFunctor interface ------------

// Moified huber for binary classification with y in {-1, 1}.
//               |max(0, 1- yp)^2 for yp >= -1
// loss(p, y) =  |
//               |-4yp otherwise
template<typename T>
struct ModifiedHuberLoss : public LossFunctor<T> {
  bool IsClassification() const { return true; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;
};

// Hinge loss for binary classification tasks with y in {-1, 1}
// loss(p, y) = max(0, param - p * y).
template<typename T>
struct HingeLoss : public LossFunctor<T> {
  HingeLoss();
  explicit HingeLoss(const T param);

  HingeLoss(const HingeLoss& src) = delete;
  HingeLoss& operator=(const HingeLoss& rhs) = delete;

  bool IsClassification() const { return true; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;

 private:
  T param_;
};

// Squared Hinge loss for binary classification tasks with y in {-1, 1}
// loss(p, y) = max(0, param - py)^2
template<typename T>
struct SquaredHingeLoss : public LossFunctor<T> {
  SquaredHingeLoss();
  explicit SquaredHingeLoss(const T param);

  SquaredHingeLoss(const SquaredHingeLoss& src) = delete;
  SquaredHingeLoss& operator=(const SquaredHingeLoss& rhs) = delete;

  bool IsClassification() const { return true; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;

 private:
  T param_;
};

// Logistic regression loss for binary classification with y in {-1, 1}.
// loss(p, y) = ln(1 + e^(-py)).
template<typename T>
struct LogLoss : public LossFunctor<T> {
  bool IsClassification() const { return true; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;
};

// Squared loss traditionally used in linear regression.
// loss(p, y) = 0.5 * (p - y)^2.
template<typename T>
struct SquaredLoss : public LossFunctor<T> {
  bool IsClassification() const { return false; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;
};

// Huber regression loss.
template<typename T>
struct HuberLoss : public LossFunctor<T> {
  HuberLoss();
  explicit HuberLoss(const T param);

  HuberLoss(const HuberLoss& src) = delete;
  HuberLoss& operator=(const HuberLoss& rhs) = delete;

  bool IsClassification() const { return false; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;

 private:
  T param_;
};

// Epsilon insensitive loss for regression.
// loss(p, y) = max(0, |p - y| - epsilon)
template<typename T>
struct EpsilonInsensitiveLoss : public LossFunctor<T> {
  EpsilonInsensitiveLoss();
  explicit EpsilonInsensitiveLoss(const T epsilon);

  EpsilonInsensitiveLoss(const EpsilonInsensitiveLoss& src) = delete;
  EpsilonInsensitiveLoss& operator=(const EpsilonInsensitiveLoss& rhs) = delete;

  bool IsClassification() const { return false; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;

 private:
  T epsilon_;
};

// Squared epsilon insensitive loss for regression
// loss = max(0, |p - y| - epsilon)^2.
template<typename T>
struct SquaredEpsilonInsensitiveLoss : public LossFunctor<T> {
  SquaredEpsilonInsensitiveLoss();
  explicit SquaredEpsilonInsensitiveLoss(const T epsilon);

  SquaredEpsilonInsensitiveLoss(const SquaredEpsilonInsensitiveLoss& src) = delete;  // NOLINT
  SquaredEpsilonInsensitiveLoss& operator=(const SquaredEpsilonInsensitiveLoss& rhs) = delete;  // NOLINT

  bool IsClassification() const { return false; }

  T operator()(const T p, const T y) const;
  T FirstDerivative(const T p, const T y) const;

 private:
  T epsilon_;
};
}  // namespace sgd
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SGD_LOSS_H_
