// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INTERNAL_SAPIEN_UTILITY_WEIGHT_VECTOR_H_
#define INTERNAL_SAPIEN_UTILITY_WEIGHT_VECTOR_H_

#include <stddef.h>  /* size_t */

namespace sapien {
namespace internal {

// WeightVector is the class template that wraps around an existing 'vector'
// object (float or double C array).
// The reason for this class to exist is this:
//      We want to speed up the weight updaing process in online learning
//      (a.k.a SGD - Stochastic Gradient Descent), particularly to speed up
//      these operations:
//          - scaling a vector by a scalar, and
//          - updating weight average.
//
// Refer to the paper [1] for more details about tips and trics used.
// [1] - http://research.microsoft.com/pubs/192769/tricks-2012.pdf.
template<typename T>
class WeightVector {
 public:
  const size_t n_elem;

  // WARNING: It's the caller's responsibility to ensure that
  // two arrays weight and average_weight have the same size!
  WeightVector(const size_t N, T* weight, T* average_weight = NULL);
  ~WeightVector();

  // We explicitly 'delete' default constructor, copy constructor, and
  // assignment operator
  WeightVector() = delete;
  WeightVector(const WeightVector&) = delete;
  WeightVector& operator=(const WeightVector&) = delete;

  // weight <- weight + alpha * x.
  //
  // WARNING: x must be the same size as weight (equal to n_elem).
  void PlusAX(const T alpha, const T* x);

  // Add alpha * x to the average_weight. n_iter is the current iteration
  // count. See [1] for more details.
  //
  // WARNING: x must be the same size as weight (equal to n_elem).
  void AveragePlusAX(const size_t n_iter, const T alpha, const T* x);

  // Computes dot product between weight and x.
  //
  // WARNING: x must be the same size as weight (equal to n_elem).
  T Dot(const T* x) const;

  // Scale weight by scalar alpha.
  void Scal(const T alpha);

  // Reset scale to 1.0
  void Reset();

  // Return the current scale
  T scale() const;

  // Compute l2-norm of the weight vector.
  T nrm2() const;

  // Element accessors
  T& operator()(const size_t i);
  const T& operator()(const size_t i) const;

  T& operator[](const size_t i) { return this->operator()(i); }
  const T& operator[](const size_t i) const { return this->operator()(i); }

 private:
  T* weight_;
  T* average_weight_;

  // Two scalar used by the trick in updating average weight as described
  // in [1].
  T alpha_;
  T beta_;

  // Current scale
  T scale_;
};
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_WEIGHT_VECTOR_H_
