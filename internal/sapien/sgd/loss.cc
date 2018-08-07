// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#include <cmath>

#include "sapien/sgd/loss.h"

namespace sapien {
namespace sgd {

// Modified huber loss -----------------------------------------------------

// Evaluate the loss at the prediction p w.r.t the ground truth y.
template<typename T>
T ModifiedHuberLoss<T>::operator()(const T p, const T y) const {
  T z = p * y;
  if (z >= T(1.0)) {
    return T(0.0);
  } else if (z >= T(-1.0)) {
    return (T(1.0) - z) * (T(1.0) - z);
  } else {
    return T(-4.0) * z;
  }
}

// First derivative of the loss function w.r.t prediction p.
template<typename T>
T ModifiedHuberLoss<T>::FirstDerivative(const T p, const T y) const {
  T z = p * y;
  if (z >= T(1.0)) {
    return T(0.0);
  } else if (z >= T(-1.0)) {
    return T(-2.0) * y * (T(1.0) - z);
  } else {
    return T(-4.0) * y;
  }
}

// We explicitly instantiate ModifiedHuberLoss for float and double types
template struct ModifiedHuberLoss<float>;
template struct ModifiedHuberLoss<double>;

// Hinge loss ---------------------------------------------------------------

template<typename T>
HingeLoss<T>::HingeLoss() : param_(1.0) {}

template<typename T>
HingeLoss<T>::HingeLoss(const T param) : param_(param) {}

template<typename T>
T HingeLoss<T>::operator()(const T p, const T y) const {
  T z = p * y;
  if (z <= param_) {
    return param_ - z;
  }
  return T(0.0);
}

template<typename T>
T HingeLoss<T>::FirstDerivative(const T p, const T y) const {
  T z = p * y;
  if (z <= param_) {
    return -y;
  }
  return T(0.0);
}

// We explicitly instantiate HingeLoss for float and double types.
template struct HingeLoss<float>;
template struct HingeLoss<double>;

// Squared hinge loss ------------------------------------------------------

template<typename T>
SquaredHingeLoss<T>::SquaredHingeLoss() : param_(1.0) {}

template<typename T>
SquaredHingeLoss<T>::SquaredHingeLoss(const T param) : param_(param) {}

template<typename T>
T SquaredHingeLoss<T>::operator()(const T p, const T y) const {
  T z = param_ - p * y;
  if (z >= T(0.0)) {
    return z * z;
  }
  return T(0.0);
}

template<typename T>
T SquaredHingeLoss<T>::FirstDerivative(const T p, const T y) const {
  T z = param_ - p * y;
  if (z >= T(0.0)) {
    return T(2.0) * (-y) * z;
  }
  return T(0.0);
}

// We explicitly instantiate SquaredHingeLoss for float and double types.
template struct SquaredHingeLoss<float>;
template struct SquaredHingeLoss<double>;

// Logistic regression loss ------------------------------------------------
template<typename T>
T LogLoss<T>::operator()(const T p, const T y) const {
  T z = p * y;
  if (z > T(18)) {
    return std::exp(-z);
  } else if (z < T(-18)) {
    return -z;
  } else {
    return std::log(T(1.0) + std::exp(-z));
  }
}

template<typename T>
T LogLoss<T>::FirstDerivative(const T p, const T y) const {
  T z = p * y;
  if (z > T(18.0)) {
    return (-y) * std::exp(-z);
  } else if (z < T(-18.0)) {
    return -y;
  } else {
    return (-y) / ( T(1.0) + std::exp(z));
  }
}

template struct LogLoss<float>;
template struct LogLoss<double>;

// Squared loss ----------------------------------------------------------

template<typename T>
T SquaredLoss<T>::operator()(const T p, const T y) const {
  return T(0.5) * (p - y) * (p - y);
}

template<typename T>
T SquaredLoss<T>::FirstDerivative(const T p, const T y) const {
  return (p - y);
}

template struct SquaredLoss<float>;
template struct SquaredLoss<double>;

// Huber regression loss --------------------------------------------------

template<typename T>
HuberLoss<T>::HuberLoss() : param_(0.1) {}

template<typename T>
HuberLoss<T>::HuberLoss(const T param) : param_(param) {}

template<typename T>
T HuberLoss<T>::operator()(const T p, const T y) const {
  T r = p - y;
  T abs_r = std::fabs(r);
  if (abs_r <= param_) {
    return T(0.5) * r * r;
  } else {
    return (param_ * abs_r) - (T(0.5) * param_ * param_);
  }
}

template<typename T>
T HuberLoss<T>::FirstDerivative(const T p, const T y) const {
  T r = p - y;
  T abs_r = std::fabs(r);
  if (abs_r <= param_) {
    return r;
  } else if (r >= T(0.0)) {
    return param_;
  } else {
    return -param_;
  }
}

template struct HuberLoss<float>;
template struct HuberLoss<double>;

// Epsilon insensitive loss ------------------------------------------------

template<typename T>
EpsilonInsensitiveLoss<T>::EpsilonInsensitiveLoss() : epsilon_(0.0) {}

template<typename T>
EpsilonInsensitiveLoss<T>::EpsilonInsensitiveLoss(const T epsilon)
    : epsilon_(epsilon) {}

template<typename T>
T EpsilonInsensitiveLoss<T>::operator()(const T p, const T y) const {
  T ret = std::fabs(y - p) - epsilon_;
  return (ret > T(0.0)) ? ret : T(0.0);
}

template<typename T>
T EpsilonInsensitiveLoss<T>::FirstDerivative(const T p, const T y) const {
  if ((y - p) > epsilon_) {
    return T(-1.0);
  } else if ((p - y) > epsilon_) {
    return T(1.0);
  } else {
    return T(0.0);
  }
}

template struct EpsilonInsensitiveLoss<float>;
template struct EpsilonInsensitiveLoss<double>;

// Squared epsilon insensitive loss ----------------------------------------

template<typename T>
SquaredEpsilonInsensitiveLoss<T>::SquaredEpsilonInsensitiveLoss()
    : epsilon_(0.0) {}

template<typename T>
SquaredEpsilonInsensitiveLoss<T>::SquaredEpsilonInsensitiveLoss(const T epsilon)
    : epsilon_(epsilon) {}

template<typename T>
T SquaredEpsilonInsensitiveLoss<T>::operator()(const T p, const T y) const {
  T ret = std::fabs(y - p) - epsilon_;
  return (ret > T(0.0)) ? ret * ret : T(0.0);
}

template<typename T>
T SquaredEpsilonInsensitiveLoss<T>::FirstDerivative(const T p,
                                                    const T y) const {
  T z = y - p;
  if (z > epsilon_) {
    return T(-2.0) * (z - epsilon_);
  } else if (z < -epsilon_) {
    return T(-2.0) * (z + epsilon_);
  } else {
    return T(0.0);
  }
}

template struct SquaredEpsilonInsensitiveLoss<float>;
template struct SquaredEpsilonInsensitiveLoss<double>;
}  // namespace sgd
}  // namespace sapien

