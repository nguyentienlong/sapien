// Copyright 2018.

#include "sapien/utility/weight_vector.h"
#include "sapien/internal/sapien_math.h"

namespace sapien {
namespace internal {

template<typename T>
WeightVector<T>::WeightVector(const size_t N, T* weight, T* average_weight)
    : n_elem(N),
      weight_(weight),
      average_weight_(average_weight),
      scale_(1.0),
      alpha_(0.0),
      beta_(1.0) {
}

template<typename T>
WeightVector<T>::~WeightVector() {
  if (scale_ != 1.0) { Reset(); }
  weight_ = NULL;
  average_weight_ = NULL;
}

template<typename T>
void
WeightVector<T>::PlusAX(const T alpha, const T* x) {
  sapien_axpy(n_elem, alpha / scale_, x, weight_);
}

template<typename T>
void
WeightVector<T>::AveragePlusAX(const size_t n_iter, const T alpha,
                               const T* x) {
  if (average_weight_ == NULL) { return; }
  T mu = 1.0 / static_cast<T>(n_iter);
  sapien_axpy(n_elem, alpha_ * (-alpha / scale_),
              x, average_weight_);
  if (n_iter > 1) {
    beta_ /= (1.0 - mu);
  }
  alpha_ += (mu * beta_ * scale_);
}

template<typename T>
T WeightVector<T>::Dot(const T* x) const {
  return scale_ * sapien_dot(n_elem, weight_, x);
}

template<typename T>
T WeightVector<T>::nrm2() const {
  return scale_ * sapien_nrm2(n_elem, weight_);
}

template<typename T>
void WeightVector<T>::Scal(const T alpha) {
  scale_ *= alpha;
  if (scale_ < 1e-9) { Reset(); }
}

template<typename T>
void WeightVector<T>::Reset() {
  //! Reset average weight vector
  if (average_weight_ != NULL) {
    sapien_axpy(n_elem, alpha_, weight_, average_weight_);
    sapien_scal(n_elem, 1/beta_, average_weight_);
    alpha_ = 0.0;
    beta_ = 1.0;
  }

  //! Reset weight vector
  sapien_scal(n_elem, scale_, weight_);
  scale_ = 1.0;
}

template<typename T>
T WeightVector<T>::scale() const { return scale_; }

template<typename T>
T& WeightVector<T>::operator()(const size_t i) {
  return *(weight_ + i);
}

template<typename T>
const T& WeightVector<T>::operator()(const size_t i) const {
  return *(weight_ + i);
}

template class WeightVector<float>;
template class WeightVector<double>;

}  // namespace internal
}  // namespace sapien
