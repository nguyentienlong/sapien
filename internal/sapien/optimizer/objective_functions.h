// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Interfaces for various objective functions.

#ifndef INTERNAL_SAPIEN_OPTIMIZER_OBJECTIVE_FUNCTIONS_H_
#define INTERNAL_SAPIEN_OPTIMIZER_OBJECTIVE_FUNCTIONS_H_

namespace sapien {
namespace internal {

// First order function.
// The FirstOrderFunction class models any differentiable function f:
//
//  f : I --------------------> R, in which I is a subset of R^n
//      x = (x1, x2, .., xn) -> f(x) = f(x1, x2, .., xn)
class FirstOrderFunction {
 public:
  FirstOrderFunction() {}
  virtual ~FirstOrderFunction() {}

  // Returns the number of variables
  virtual int n_variables() const = 0;

  // Evaluate this function at the given point x.
  // Note that, it is the caller's responsibility to make sure that the size
  // of x is equal to n_variables.
  virtual double operator()(const double* x) const = 0;

  // Evaluate the gradient of this function at the given point x.
  // Note that, it is the caller's responsibility to make sure that the size
  // of x (as well as gradient) is equal to n_variables.
  virtual void Gradient(double* gradient, const double* x) const = 0;
};
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_OPTIMIZER_OBJECTIVE_FUNCTIONS_H_
